from __future__ import absolute_import, division, print_function, unicode_literals
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from flask import Flask, request, jsonify
import spacy
import pandas as pd
from pymongo import MongoClient
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import json


from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification

app = Flask(__name__)
app.config['Debug'] = True

model = spacy.load('NER_SPACY_RED_CANOPY_V1')
client = MongoClient('mongodb://datastore:27017',connect=False)

db = client.RedCanopy

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

  print('Tokens Generated')
  return (input_ids,attention_masks,time.time() - gen_time)

def preprocess(data):
  row_strs_s = []
  for i,row in data.iterrows():
    row_s = str(row['symptoms']).replace('"','').strip()
    row_s = " ".join(row_s.split())
    row_strs_s.append(row_s)
  return pd.Series(row_strs_s)

@app.route('/getClassificationsemb', methods=['GET', 'POST'])
def load_model():
  start_time = time.time()
  if request.method == 'POST':
      try:
        cat_colss = ['gender', 'start', 'onset','character', 'progression', 'duration', 'severity',     'triggered_by','relieved_by', 'lead_symptoms', 'burping', 'abdomen_bloating',
       'food_regurgitation', 'anorexia', 'tenderness', 'diarrhea', 'chestpain',
       'hoarseness', 'odynophagia', 'weightloss', 'dysphagia', 'chronic_cough',
       'bleeding', 'feeling_full_after_small_amount_of_food', 'heartburn',
       'joint_soreness', 'abscess_anus_area', 'purulent_drainage_in_anus',
       'non_healing__lump', 'prolapse_of_anal_tissue',
       'inability_to_pass_flatus', 'change_in_stool_frequency',
       'abdomen_cramps', 'headache', 'upper_abdomen_ache', 'coughing',
       'presyncope', 'dizziness', 'diaphoresis', 'belching', 'jaw_pain',
       'indigestion', 'fainting', 'anxiety', 'palpitation', 'cardiax_arrest',
       'sweatiness', 'pleuritic_chestpain', 'wheezing', 'swollen_calf',
       'couging_blood', 'worse_with_deep_breathing', 'food_stuck_in_throat',
       'difficulty_speaking', 'loss_of_vision', 'leg_pain', 'leg_paralysis',
       'paralysis_on_one_side', 'pain_with_movement', 'focal_tenderness',
       'difficulty_swallowing', 'muscle_ache', 'altered_mental_status',
       'loss_of_appetite', 'history', 'frequency', 'radiation',
       'location(where)', 'mouth_or_gum_ulcer', 'weakness', 'fatigue',
       'weakness_and_fatigue', 'fatigue_malaise', 'cough_with_sputum',
       'pre_syncope_Fainting', 'shortness_of_breath', 'abdomen_pain',
       'blood_in_stool', 'sweating', 'nausea_or_vomiting', 'fever_or_chill',
       'symptom_in_the_past', 'difficulty_walking']

        cont_colss = ["age"]
        data_dic = {}
        with open('label_dic.json') as json_f:
            data_dic = json.load(json_f)
        data = request.files['data_excel']
        data_full=pd.read_excel(data)

        for cat in cat_colss:
            data_full[cat] = data_full[cat].str.lower().str.strip()
            

        for cont in cont_colss:
            data_full[cont] = data_full[cont].replace('na', np.NAN)
            data_full[cont] = data_full[cont].fillna(data_full[cont].median())
            data_full[cont] = data_full[cont].astype('float')
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

        df_full=convert_test(data_full,data_dic,cat_colss)
        for cat in cat_colss:
          df_full[cat] = df_full[cat].astype('int')

        cats = df_full[cat_colss].to_numpy()
        conts = data_full['age'].to_numpy()

        emb_sizes = [(3, 2), (15, 8), (4, 2), (30, 15), (8, 4), (36, 18), (8, 4), (33, 17), (20, 10), (6, 3), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (1, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (11, 6), (4, 2), (25, 13), (25, 13), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (4, 2), (2, 1), (2, 1), (2, 1), (6, 3), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1)]
        cats_tensor=torch.tensor(cats,dtype=torch.int64)
        conts_tensor=torch.tensor(conts,dtype=torch.float64)
        torch.manual_seed(9)
        selfembeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_sizes])
        #print(selfembeds)
        embeddingz=[]
        for j,e in enumerate(selfembeds):
          
          embeddingz.append(e(cats_tensor[:,j]))

        X_cat = torch.cat(embeddingz, axis=1)
        print(X_cat.shape)
        #print(conts_tensor.reshape(-1,1))    
        X_total=torch.cat([conts_tensor.reshape(-1,1).float(),X_cat],axis=1)
        print(X_total.shape)
        model_torch_reload=Tabular_Modelemb(emb_sizes,X_total.shape[1],30,[256,128,64,30],p=0.2)
        st_d = torch.load("model_iteration1_with_embedding_V1.pth")
        model_torch_reload.load_state_dict(st_d)
        model_torch_reload.eval()
        #print(model_torch_reload)
        with torch.no_grad():
          y_val_reload = model_torch_reload(X_total)

        #print(y_val_reload)

        class_names = ['Angina ', 'Aortic Dissection', 'Costochondritis',
                       'Esophageal Perforation', 'Esophageal Spasm ',
                       'Myocardial Infarction ', 'Pericarditis ', 'Pleuritis ',
                       'Pneumonia', 'Pneumothorax', 'Pulmonary Embolism ', 'Rib Fracture',
                       'acute_apendicitis', 'acute_diverticulitis', 'anal_fissure',
                       'anal_fistula', 'bowel_obstruction', 'cholecystitis',
                       'chrons_disease', 'colon_cancer', 'esophageal_malignancy_cancer',
                       'external_thrombosed_hemorrhoid', 'gallstone', 'gastic_malignancy',
                       'gastroenteritis', 'gerd', 'hiatus_hernia', 'internal_hemorrhoid',
                       'peptic_ulcer', 'perirectal_abscess']

        y_val_names = []

        for i in range(len(y_val_reload)):
            for j, v in enumerate(class_names):
                if y_val_reload[i].argmax().item() == j:
                    y_val_names.append(v)

        db.classifications.insert_many({'Preds':y_val_names})
        return jsonify({"success": "Data inserted", 'Execution Time': time.time()-start_time})
        #return y_val_names
      except Exception as e:
        return jsonify({"Exception occured": f'{sys.exc_info()[0]} -- {str(e)}'})
  else:
    return jsonify({"success": "method not allowed"})

@app.route('/getAnnotations', methods=['GET', 'POST'])
def getAnn():
    if request.method == 'POST':
        try:
            start_time = time.time()
            data_csv = request.files['data']
            #data = pd.read_csv(data_csv, names=['Data'], header=None)
            #data = pd.read_csv(data_csv, header=None)
            data = pd.read_excel(data_csv)
            data = preprocess_data(data)          
            data_li = []
            for i, row in data.iterrows():
                data_dic = {'Age': 0, 'Gender': 'none', 'Symptoms': 'none', 'Duration': 'none', 'Location': 'none', 'Onset': 'none', 'Tests': 'none', 'History': 'none', 'Frequency': 'none', 'Pain': 'none',
                        'Progression': 'none', 'TrigerredBy': 'none', 'RelievedBy': 'none', 'Severity': 'none', 'When': 'none', 'Sensation': 'none', 'Character': 'none', 'Difficulty': 'none', 'LeadSymptom': 'none'}
                doc = model(row['Data'])
                for ent in doc.ents:
                    if ent.label_ == 'Age' or ent.label_ == 'Gender':
                        if ent.label_ == 'Age':
                            data_dic[ent.label_] = ent.text
                        elif ent.label_ == 'Gender':
                            if ent.text == 'f' or ent.text == 'm':
                                data_dic[ent.label_] = ent.text
                    elif isinstance(data_dic[ent.label_], list):
                        data_dic[ent.label_].append(ent.text)
                    else:
                        data_dic[ent.label_] = [ent.text]
                data_li.append(data_dic)
            db.EMRS_Data_csv.insert_many(data_li)
            return jsonify({"success": "Data inserted", 'Execution Time': time.time()-start_time})
        except Exception as e:
            return jsonify({"Exception occured":str(e)})
    else:
        return jsonify({"success": "method not allowed"})

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
  print('Sucess')
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

@app.route('/getClassifications', methods=['GET', 'POST'])
def getCls():
  if request.method == 'POST':
    try:

      start_time = time.time()
      model_output_labels = []
      max_seq_length = 128
      data_csv = request.files['data']
      data = pd.read_csv(data_csv)
      data['symptoms'] = preprocess(data)
      train_text = data['symptoms'].tolist()
      train_text = [' '.join(t.split()[0:max_seq_length]) for t in train_text]
      train_text = np.array(train_text, dtype=object)[:, np.newaxis] 

      input_ids,input_masks,gen_time =gen_id_mask(data)
      # model_pybert_new_2 = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=13)
      # st_d=torch.load("bert_pytorch_13_v5.pt")
      # model_pybert_new_2.load_state_dict(st_d)
      print(torch.tensor(input_ids).shape,torch.tensor(input_masks).shape)

      model = torch.load("Bert_1_v2.pt")
      train_inputs = torch.tensor(input_ids,dtype=torch.long)
      train_masks = torch.tensor(input_masks,dtype=torch.long)
      #train_data = TensorDataset(train_inputs, train_masks)
      
      batch_size = 64
      #train_dataloader = DataLoader(train_data, batch_size=batch_size)
      target_names=['acute_appendicitis',
                        'acute_cholecystitis',
                       'acute_pancreatitis',
                       'bowel_obstruction',
                       'choledocholithiasis',
                       'congestive_heart_failure',
                       'copd',
                       'diverticulitis',
                       'myocardial_Infarction',
                       'peptic_ulcer_disease',
                       'pneumonia',
                       'pneumothorax',
                       'pulmonary_embolism']
      pred_time = time.time()
      with torch.no_grad():      
        i = 0
        # Forward pass, calculate logit predictions
        pass_time = time.time()
        logits = model(train_inputs, token_type_ids=None, attention_mask=train_masks)
        m_p_time = time.time() - pass_time
        #print(logits.shape)
        pred_flat_1 = np.argmax(logits, axis=1).flatten()
        pred_flat_1=pred_flat_1.tolist()
        m_n = [target_names[i] for i in pred_flat_1]
        model_output_labels.append({str(i) : m_n})
        i += 1
      preds_time = time.time() - pred_time
      in_time = time.time()
      db.classification_classes.insert_many(model_output_labels)
      insert_time = time.time() - in_time
      return jsonify({"success": "Data inserted", 'Execution Time': time.time()-start_time , 'Token time' : gen_time, 'prediction time':preds_time,'insert time':insert_time,'Pass through model':m_p_time})
    except Exception as e:
      return jsonify({"Exception occured":str(e)})
  else:
    return jsonify({"success": "method not allowed"})



if __name__ == "__main__":
    app.run(host='0.0.0.0')
