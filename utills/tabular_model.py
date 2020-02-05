import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_pretrained_bert import (BertAdam, BertForSequenceClassification,
                                     BertTokenizer)

class Tabular_Modelemb(nn.Module):

  def __init__(self,emb_szs,n_total,out_size,layers,p=0.2):
    super().__init__()



    #Set up a dropout function for the embeddings with torch.nn.Dropout() The default p-value=0.5
    self.emb_drop = nn.Dropout(p)

    #Set up a normalization function for the continuous variables with torch.nn.BatchNorm1d()
    #self.bn_cont = nn.BatchNorm1d(n_total)
    

    # Set up a sequence of neural network layers where each level includes a Linear function, an activation function (we'll use ReLU), a normalization step, and a dropout layer. We'll combine the list of layers with torch.nn.Sequential()
    # self.bn_cont = nn.BatchNorm1d(n_cont)
    layerlist = []
    
    n_in = n_total

    for i in layers:
        layerlist.append(nn.Linear(n_in,i))
        layerlist.append(nn.ReLU(inplace=True))
        layerlist.append(nn.BatchNorm1d(i))
        layerlist.append(nn.Dropout(p))
        n_in = i

    layerlist.append(nn.Linear(layers[-1],out_size))

    self.layers = nn.Sequential(*layerlist,nn.Softmax())


  def forward(self, x_total):

    x = self.emb_drop(x_total)

    #x = self.bn_cont(x)
    x = self.layers(x)
    return x
