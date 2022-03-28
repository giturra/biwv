import torch
import numpy as np
from streamdataloader import TweetStreamLoader
from iwcm import WordContextMatrix
from iglove import IGlove
from nltk import word_tokenize


fn = "weet.txt"
bat_size = 256 
buff_size = 2048
emp_ldr = TweetStreamLoader(fn, bat_size, buff_size, shuffle=False)
igv = IGlove(100, 10000, 20000, 3, device=torch.device('cpu'))

for (b_idx, batch) in enumerate(emp_ldr):
  igv.learn_many(batch)
  
    
emp_ldr.fin.close()
