****************************************************************
cell: 01:
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics
from sklearn.model_selection import train_test_split
import transformers
from transformers import AdamW

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

import warnings
warnings.filterwarnings("ignore")

********************************************************************

cell02:

t0 = torch.randn(2, 2, device=xm.xla_device()) #creating a tensor and sending it to the TPU
t1 = torch.randn(2, 2, device=xm.xla_device()) #creating a tensor and sending it to the TPU
print(t0 + t1) # As both tensors are now on the same device  i.e same TPU core we can perform any calculation on them like addition
print(t0.mm(t1)) # Matrix Multiplication

********************************************************************

cell03:

l_in = torch.randn(10, device=xm.xla_device())
linear = torch.nn.Linear(10, 20).to(xm.xla_device())
l_out = linear(l_in) #NOTE THAT THE TENSOR AND MODEL BOTH BE SENT TO THE DEVICE AS WE DID WITH GPUS , THEN ONLY we CAN PERFORM ANY OPERATION
print(l_out)


********************************************************************

cell04:

class config:
    
    MAX_LEN = 224
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 8
    EPOCHS = 1
    MODEL_PATH = "model.bin"
    TRAINING_FILE = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv'
    TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case =True)

********************************************************************

cell07:
class BERTDataset(torch.utils.data.Dataset):
    def __init__(self,text,target):
        self.text = text
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN 

    def __len__(self):
        return len(self.text)

    def __getitem__(self,idx):
        text  = str(self.text[idx])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens = True,
            max_length = self.max_len,
            pad_to_max_length = True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            'ids': torch.tensor(ids,dtype=torch.long),
            'mask': torch.tensor(mask,dtype=torch.long),
            'targets': torch.tensor(self.target[idx],dtype=torch.long)
        }

********************************************************************