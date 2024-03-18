import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, fbeta_score
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json

def get_metric_dict(labels, preds):
    acc = accuracy_score(labels, preds)
    bac = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    prec = precision_score(labels, preds, average='weighted')
    rec = recall_score(labels, preds, average='weighted')
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
    return df

def get_join(doc_ids, preds, gts):
    new_slim_preds = []
    clean_doc_ids = {}
    ground_truths = []
    for i, idd in enumerate(doc_ids):
        if '_' in idd:
            idd = idd[:idd.find('_')]

        p = preds[i]
        gt = gts[i]
        slim_pred = {
            'doc_id': idd,
            'prediction': p,
            'ground_truth': gt,
        }
        if idd not in clean_doc_ids.keys():
            clean_doc_ids[idd] = slim_pred
            continue
        if p == 0:
            continue
        clean_doc_ids[idd] = slim_pred
    return clean_doc_ids

def fix_preds(gens):
    sm = 0
    sn = 0
    new_preds = []
    for g in gens:
        x = g.split('\n')
        ind = 0
        for i, bi in enumerate(x):
            if 'The text does' in bi:
                ind = i

        x = x[ind]
        if 'The text does not' in x:
            sm += 1
            new_preds.append(0)
        elif 'The text does contain' in x:
            sn +=1
            new_preds.append(1)

    #print(sm)
    #print(sn)
    return new_preds



def run_evaluation(name):
    metrics_data = {}
    readjson = get_results_json('mistending.json')
    doc_ids = readjson['doc_id'].to_list()
    preds = readjson['prediction'].to_list()
    gts = readjson['ground_truth'].to_list()

    gens = readjson['full_response'].to_list()
    #print(gens[0])

    preds = fix_preds(gens)

    cln = get_join(doc_ids, preds, gts)
    cln = (list(cln.values()))
    cln_df = pd.DataFrame(cln)
    
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
