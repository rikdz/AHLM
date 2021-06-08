import torch
from torch import nn
from torch.nn.functional import gumbel_softmax, softmax
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Model, AdamW
from tqdm import tqdm
from os import listdir
import pytorch_lightning as pl
import re
from math import exp

class pplRegress(pl.LightningModule):

  def __init__(self):
    super().__init__()
    model_id = 'gpt2'
    self.gpt2 = GPT2Model.from_pretrained(model_id)
    self.regressor = nn.Sequential(
      nn.Linear(768,256),
      nn.ReLU(),
      nn.Linear(256,1)
    )
  
  def forward(self,x):
    op = self.gpt2(x,return_dict=True)
    eos = op.last_hidden_state[:,-1,:]
    eos = self.regressor(eos)
    return eos.squeeze(1).float()

  def training_step(self,batch,batch_idx):
    x,y = batch
    x = x.squeeze(1)
    y = y.float()
    yhat = self.forward(x)
    loss = F.mse_loss(yhat,y)
    self.log('train_loss',loss)
    return loss

  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=1e-5)
    return optimizer

device = 'cuda'
ranker_fn = None
ranker= pplRegress.load_from_checkpoint('models/wiki_gpt_ranker.ckpt').to(device)
ranker.eval()
    
model_id = 'gpt2-large'
# Another datapoint
#model_id = 'gpt2'
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
breaktok = tokenizer("\n\n",return_tensors="pt")['input_ids']
gpt2lm = GPT2LMHeadModel.from_pretrained(model_id).to(device)

from nlp import load_dataset
files = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
docs = [] 
d = []
p = re.compile(" = [^=]")
for x in files['text']:
  if p.match(x):
    docs.append('\n'.join(d))
    d = []
  else:
    d.append(x)

docs.append('\n'.join(d))
docs = [tokenizer(x,return_tensors="pt") for x in docs if x.strip()!='']

compressed = 512
chunk = 64
max_length = gpt2lm.config.n_positions
n_chunks = int((max_length-compressed)/chunk)
stride = 256

chunky_doc = []
basic_doc = []

basic_tok = []
chunky_tok = []

for encodings in tqdm(docs):
  basic = []
  chunky = []
  for i in range(0,encodings.input_ids.size(1),stride):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, encodings.input_ids.size(1))
    trg_len = end_loc - i    # may be different from stride on last loop

    # compute p(Future | Context)
    pfc_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
    target_ids = pfc_ids.clone()
    target_ids[:,:-trg_len] = -100

    with torch.no_grad():
      outputs = gpt2lm(pfc_ids, labels=target_ids)
      pfc_log_likelihood = outputs[0] * trg_len
    basic.append(pfc_log_likelihood)
    if begin_loc<max_length:
      chunky.append(pfc_log_likelihood)
      continue

    # sort history by ranker
    torch.cuda.empty_cache()
    uncomp = end_loc-compressed
    minhist = []
    cbc_ids2 = encodings.input_ids[:,uncomp-chunk:end_loc].clone().to(device)
    # checking 1 chunk at a time:
    for j in range(chunk,uncomp,chunk//2):
      hist_ids = encodings.input_ids[:,j-chunk:j].clone()
      hist_ids[:,-1] = breaktok
      cbc_ids2[:,:chunk]=hist_ids
      with torch.no_grad():
        score = ranker(cbc_ids2)
      minhist.append((score,hist_ids))
    minhist.sort(key=lambda x:x[0])
    torch.cuda.empty_cache()
    best_chunks = torch.cat([x[1] for x in minhist[-n_chunks:]],1)
    assert(best_chunks.size(1)==compressed)
    pfc_ids[:,:compressed] = best_chunks
    with torch.no_grad():
      outputs = gpt2lm(pfc_ids, labels=target_ids)
      best_log_likelihood = outputs[0] * trg_len
    chunky.append(best_log_likelihood)
  if chunky:
    ppl = torch.exp(torch.stack(basic).sum() / end_loc)
    basic_doc.append(ppl)
    ppl = torch.exp(torch.stack(chunky).sum() / end_loc)
    chunky_doc.append(ppl)
    basic_tok.append((torch.stack(basic).sum().item(),end_loc))
    chunky_tok.append((torch.stack(chunky).sum().item(),end_loc))
#print('basic_doc',sum(basic_doc).item()/len(basic_doc))
#print('chunk_doc',sum(chunky_doc).item()/len(chunky_doc))
print("basic_tok",exp(sum([x[0] for x in basic_tok])/sum([x[1] for x in basic_tok])))
print("chunk_tok",exp(sum([x[0] for x in chunky_tok])/sum([x[1] for x in chunky_tok])))
assert(len(basic_doc)==len(chunky_doc))




