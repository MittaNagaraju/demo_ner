from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
import os
import sys
import time

from flask import Flask, jsonify, request

import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from pytorch_pretrained_bert import (BertAdam, BertForSequenceClassification,
                                     BertTokenizer)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from utills.connection import open_connection
from utills.tabular_model import Tabular_Modelemb
from utills.helper_functions import (convert_test,gen_id_mask, 
				     preprocess, preprocess_data,
                                     truncate,get_top_predictions)

app = Flask(__name__)
app.config['Debug'] = True

con = open_connection()
db = con.RedCanopy


@app.route('/getAnnotations', methods=['GET', 'POST'])
def getAnn():
    if request.method == 'POST':
        model = spacy.load('NER_SPACY_RED_CANOPY_V1')
        start_time = time.time()
        data_csv = request.files['data']
        data = pd.read_excel(data_csv)
        data = preprocess_data(data)
        data_li = []
        for i, row in data.iterrows():
            data_dic = {'Age': 0, 'Gender': 'none', 'Symptoms': 'none', 'Duration': 'none', 
                        'Location': 'none', 'Onset': 'none', 'Tests': 'none', 'History': 'none',
                        'Frequency': 'none', 'Pain': 'none','Progression': 'none', 'TrigerredBy': 'none',
                        'RelievedBy': 'none', 'Severity': 'none', 'When': 'none', 'Sensation': 'none', 
                        'Character': 'none', 'Difficulty': 'none', 'LeadSymptom': 'none'}
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
    else:
        return jsonify({"success": "method not allowed"})

if __name__ == "__main__":
    app.run(host='0.0.0.0')
