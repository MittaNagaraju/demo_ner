import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_pretrained_bert import (BertAdam, BertForSequenceClassification,
                                     BertTokenizer)
def convert_test(df, data_dic, cat_colss):
    df_cols = {}
    for col in cat_colss:
        m_list = []
        strlist_col = df[col].str.lower().str.strip()
        for i, v in enumerate(strlist_col):
            if v not in data_dic[col].values():
                m_list.append(99)
            for i, val in data_dic[col].items():
                if val == v:
                    m_list.append(i)
        df_cols[col] = m_list
    df1_test = pd.DataFrame.from_dict(df_cols)
    return df1_test

def gen_id_mask(data):
  gen_time = time.time()
  symptoms = data.symptoms.values

  # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
  symptoms = ["[CLS] " + symptom + " [SEP]" for symptom in symptoms]

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

  tokenized_texts = [tokenizer.tokenize(symptom) for symptom in symptoms]
  MAX_LEN = 256
  input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
  input_ids = truncate(input_ids,MAX_LEN)
  attention_masks = []

  # Create a mask of 1s for each token followed by 0s for padding
  for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)

  return (input_ids,attention_masks,time.time() - gen_time)

def preprocess(data):
  row_strs_s = []
  for i,row in data.iterrows():
    row_s = str(row['symptoms']).replace('"','').strip()
    row_s = " ".join(row_s.split())
    row_strs_s.append(row_s)
  return pd.Series(row_strs_s)

def truncate(tokenized_text,max_len):
  truncated_text = []
  for text in tokenized_text:
    if len(text) > max_len:
      truncated_text.append(text[:max_len])
    elif len(text) < max_len:
      vi = max_len - len(text)
      for i in range(vi):
        text.append(0)
      truncated_text.append(text)
  return np.array(truncated_text)

def preprocess_data(data):
  data = data.loc[:,['Data']]
  row_strs = []
  for i,row in data.iterrows():
    row = row['Data']
    row = row.replace('"','')
    row = " ".join(row.split())
    row = row.strip()
    row_strs.append(row)
  data.loc[:,'Data'] = pd.Series(row_strs)
  return data
def get_top_predictions(logi_lis,lim = 5):
  target_names=['acute_appendicitis','acute_cholecystitis','acute_pancreatitis','bowel_obstruction',
                'choledocholithiasis','congestive_heart_failure','copd','diverticulitis','myocardial_Infarction',
                'peptic_ulcer_disease','pneumonia','pneumothorax','pulmonary_embolism']
  preds = []
  for i,lis in enumerate(logi_lis):
    print('list',i)
    m_p = []
    lis = np.array(lis)
    ind = np.argpartition(lis, -lim)[-lim:]
    ind_or = ind[np.argsort(lis[ind])]
    t_pred = ind_or[::-1]
    preds.append([{target_names[v] : str("{0:.2f}".format(float(lis[v])*10))+'%'} for i,v in enumerate(t_pred)])
  return preds