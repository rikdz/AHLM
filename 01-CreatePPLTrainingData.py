import sys 
import re
from os import listdir
from pathlib import Path

import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Model
from tqdm import tqdm

device = 'cuda'
model_id = 'gpt2-large'
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
breaktok = tokenizer("\n\n",return_tensors="pt")['input_ids']
gpt2lm = GPT2LMHeadModel.from_pretrained(model_id).to(device)

def tok_doc(dat):
  sent_text = []
  idxs = []
  if dat["grobid_parse"]:
    if "body_text" in dat["grobid_parse"]:
      for b in dat["grobid_parse"]["body_text"]:
        sent_text.extend([x+"\n" for x in b])
        sent_text.append("\n\n")
  sent_text = ''.join(sent_text)
  sent_text = tokenizer(sent_text, return_tensors="pt")
  return sent_text

from nlp import load_dataset
files = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
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

bsz = 8
compressed = 512
chunk = 64
max_length = gpt2lm.config.n_positions
stride = 256
docppl = []
dochistppl = []
IDX = 0
fn = 0

Path("wiki_chunk").mkdir(exist_ok=True)

for encodings in tqdm(docs):

  # get reps of every sent
  lls = [] 
  histlls = []
  data = []

  for i in range(compressed, encodings.input_ids.size(1),stride):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, encodings.input_ids.size(1))
    trg_len = end_loc - i    # may be different from stride on last loop
    if end_loc - begin_loc < compressed: continue

    ### these might not be needed
    input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:,:-trg_len] = -100


    # replace part of input ids with chunks from history
    baseline_ids = encodings.input_ids[:,end_loc-compressed-chunk:end_loc].to(device)
    target_ids = baseline_ids.clone()
    target_ids[:,:-trg_len] = -100

    # get baseline
    with torch.no_grad():
      outputs = gpt2lm(baseline_ids, labels=target_ids)
      base_log_likelihood = outputs[0] * trg_len

    inp_ids2 = encodings.input_ids[:,end_loc-compressed-chunk:end_loc].to(device)
    target_ids2 = inp_ids2.clone()
    target_ids2[:,:-trg_len] = -100
    b = 0
    batch = []
    chunks = []
    llhs = []
    for j in range(chunk,begin_loc,chunk//2):
      hist_ids = encodings.input_ids[:,j-chunk:j]
      input_ids2 = inp_ids2.clone()
      input_ids2[:,:chunk]=hist_ids
      input_ids2[:,chunk-1] = breaktok
      batch.append(input_ids2)
      chunks.append((j-chunk,j))
      b+=1
      if b==bsz:
        b = 0
        batch_ids = torch.cat(batch,0)
        targets = target_ids2.repeat(bsz,1)
        with torch.no_grad():
          outputs = gpt2lm(batch_ids,labels=targets)
          log_likelihood = outputs[0] * trg_len
          llhs.append(log_likelihood)
        batch = []
  
    if batch:
      batch_ids = torch.cat(batch,0)
      targets = target_ids2.repeat(batch_ids.size(0),1)
      with torch.no_grad():
        outputs = gpt2lm(batch_ids,labels=targets)
        log_likelihood = outputs[0] * trg_len
        llhs.append(log_likelihood)
    
    ### Format into training data
    ### Datapoints should have chunk, uncompressed, llh
    i = 0
    for b in llhs:
      b = b.squeeze().tolist()
      if type(b) is not list:
        b = [b]
      for x in b:
        x = base_log_likelihood - x
        x = x.item()
        uncomp = (end_loc - compressed,end_loc)
        ch = chunks[i]
        i+=1
        data.append((ch,uncomp,x))

  with open('wiki_chunk/'+str(fn)+".tsv",'w') as f:
    f.write('\n'.join(['\t'.join([str(x) for x in y]) for y in data]))
  fn+=1
