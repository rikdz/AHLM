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
from pytorch_lightning.callbacks import ModelCheckpoint

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

  def validation_step(self,batch,batch_idx):
    x,y = batch
    x = x.squeeze(1)
    y = y.float()
    yhat = self.forward(x)
    loss = F.mse_loss(yhat,y)
    self.log('val_loss',loss)
    return loss
    
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

class PPLDataset(Dataset):
  def __init__(self,data,val=False):
    if val:
      self.datapoints = data.val
    else:
      self.datapoints = data.train

  def __getitem__(self,idx):
    return self.datapoints[idx]

  def __len__(self):
    return len(self.datapoints)

class LoadData():
  def __init__(self):
    self.datadir='wiki_chunk'
    self.chunksize = 64
    self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    self.breaktok = self.tokenizer("\n\n",return_tensors="pt")['input_ids']
    self.pages = self.load_wiki_pages()
    self.train = []
    self.val = []
    for i,p in enumerate(self.pages):
      trips = self.load_ppls(i)
      if trips:
        points = self.proc_page_points(p,trips)
        if i%9==0:
          self.val.extend(points)
        else:
          self.train.extend(points)

  def proc_page_points(self,page,triples):
    points = []
    for x in triples:
      chunk, context, ppl = x
      start, end = context
      c_start,c_end = chunk
      vec = page.input_ids[:,start-self.chunksize:end].clone()
      vec[:,0:self.chunksize] = page.input_ids[:,c_start:c_end]
      vec[:,self.chunksize-1] = self.breaktok
      points.append((vec,float(ppl)))
    return points



  def load_wiki_pages(self):
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
    docs = [self.tokenizer(x,return_tensors="pt") for x in docs if x.strip()!='']
    return docs

  def load_ppls(self,idx):
    with open(self.datadir+'/'+str(idx)+'.tsv') as f:
      dat = f.readlines()
    if not dat:
      return False
    trips = []
    for l in dat:
      trips.append([eval(x) for x in l.split("\t")])
    return trips


    
model = pplRegress()
data = LoadData()
tds = PPLDataset(data)
vds = PPLDataset(data,val=True)
dataloader = DataLoader(tds,batch_size=16, shuffle=True)
valloader = DataLoader(vds,batch_size=16)
print('data loaded')
checkpoint = ModelCheckpoint(save_top_k=5,monitor="val_loss",filename="{epoch}-{step}-{val_loss}")
trainer = pl.Trainer(gpus=[1],checkpoint_callback=checkpoint)
trainer.fit(model,dataloader,valloader)
    
