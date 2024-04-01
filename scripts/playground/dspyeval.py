import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, fbeta_score
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json
import sys
import re
import email
import gensim

sys.path.append("../")
from dataset import load_sara


def full_preproc(s, tokenizer='', c_size=2048):
    def preprocess(e):
        message = email.message_from_string(e)
        clean = message.get_payload()
        clean = re.sub('\S*@\S*\s?', '', clean)
        clean = re.sub('\s+', ' ', clean)
        clean = re.sub("\'", "", clean)
        clean = gensim.utils.simple_preprocess(str(clean), deacc=True, min_len=1, max_len=100) 
        return clean

    def remove_doubles(df):
        already_exists = []
        unique_df = []
        for i, s in enumerate(df.iterrows()):
            idd = s[1].doc_id
            text = s[1].text
            sensitivity = s[1].sensitivity
            if text in already_exists:
                continue
            already_exists.append(text)
            unique_df.append({'doc_id': idd, 'text':text, 'sensitivity':sensitivity})    
        return pd.DataFrame.from_dict(unique_df)

    def main(s):
        processed_emails = [preprocess(a) for a in s.text]
        ids = s.doc_id.tolist()
        sens = s.sensitivity.tolist()
        texts = []
        for i, text in enumerate(s.text):
            new_email = ' '.join(processed_emails[i])
            texts.append(new_email)
        new_dict = {'doc_id': ids, 'text': texts, 'sensitivity':sens}
        preproc_df = pd.DataFrame.from_dict(new_dict)
        preproc_df = remove_doubles(preproc_df)
        return preproc_df

    return main(s)

def get_metric_dict(labels, preds):
    acc = accuracy_score(labels, preds)
    bac = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    prec = precision_score(labels, preds, average='weighted')
    rec = recall_score(labels, preds, average='weighted', zero_division=0)
    f2 = fbeta_score(labels, preds, average='weighted', beta=2)
    metric_dict = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'bal accuracy': bac, 'f2_score': f2}
    return metric_dict

def get_results_json(file_name):
    current_directory = os.getcwd()
    target_directory = os.path.join(current_directory, 'results')
    file_path = os.path.join(target_directory, file_name)

    with open(file_path) as json_file:
        data = json.load(json_file)

    df = pd.DataFrame(data)
    return data

def new_get_join(data):
    print(data[0])
    dddd = full_preproc(load_sara())
    collected_truths = []
    collected_docids = []
    mapp = {}
    for item in data:
        doc_id = item.get('doc_id')
        if '_' in doc_id:
            doc_id = doc_id[:doc_id.find('_')]
        truth = dddd[dddd.doc_id == doc_id].iloc[0].sensitivity
        pred = item.get('prediction')

        if doc_id not in collected_docids:
            if pred is None:
                pred = 1
            collected_docids.append(doc_id)
            collected_truths.append(truth)
            new_pred = {
                'doc_id': doc_id,
                'prediction': pred,
                'ground_truth': truth,
            }
            mapp[doc_id] = new_pred

        if pred == 1:
            new_pred = {
                'doc_id': doc_id,
                'prediction': int(pred),
                'ground_truth': int(truth),
            }
            mapp[doc_id] = new_pred

    mapp = list(mapp.values())
    return pd.DataFrame(mapp)


def run_evaluation(name):
    metrics_data = {}
    readjson = get_results_json('dspcgcot.json')
    #doc_ids = readjson['doc_id'].to_list()
    #preds = readjson['prediction'].to_list()
    #gts = readjson['ground_truth'].to_list()

    cln = new_get_join(readjson)
    #cln = (list(cln.values()))
    #cln_df = pd.DataFrame(cln)
    cln_df = cln
    TN = cln_df[(cln_df.prediction == 0) & (cln_df.ground_truth == 0)]
    TP = cln_df[(cln_df.prediction == 1) & (cln_df.ground_truth == 1)]
    FP = cln_df[(cln_df.prediction == 1) & (cln_df.ground_truth == 0)]
    FN = cln_df[(cln_df.prediction == 0) & (cln_df.ground_truth == 1)]
    print(len(TN), len(FP), len(FN), len(TP))

    cln_preds = cln_df['prediction'].to_list()
    cln_gts = cln_df['ground_truth'].to_list()

    metrics_data['dspy'] = get_metric_dict(cln_gts, cln_preds)

    print(metrics_data)
    metrics_df = pd.DataFrame.from_dict(metrics_data, orient='index')
    metric_path = f'results/metric_overview/'
    #if not os.path.exists(metric_path):
    #    os.makedirs(metric_path)
    #metrics_df.to_csv(f'{metric_path}mistchunks.csv', index=True)

run_evaluation('')
