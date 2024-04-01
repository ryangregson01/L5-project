import ir_datasets
import email
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import re
import numpy as np
import json
import os
import sys
from dataset import load_sara
from models import get_model_version
from preprocess_sara import full_preproc, clean
import gensim
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, fbeta_score, roc_auc_score


def no_reply_proc(s, tokenizer='', c_size=2048):
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

def new_get_join(data):
    collected_truths = []
    collected_docids = []
    mapp = {}
    for item in data:
        doc_id = item.get('doc_id')
        if '_' in doc_id:
            doc_id = doc_id[:doc_id.find('_')]
        truth = clean_unique_docs[clean_unique_docs.doc_id == doc_id].iloc[0].sensitivity
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

def get_results_json(mname, clean=True):
    current_directory = os.getcwd()
    target_directory = current_directory + f'/results/model_results/{mname}/'
    #print("Path to results", target_directory)
    prompt_results = os.listdir(target_directory)
    main_results = []
    prompts_and_answers = {'multi_category': 'does not',
                           'text': 'non-sensitive',
                           'pdc2': 'non-personal',
                           'cg': 'non-personal',
                           'textfew': 'non-sensitive',
                           'pdcfew': 'non-personal',
                           'cgfew': 'non-personal',
                           }
    prompt_results = os.listdir(target_directory)
    main_results = []
    for prompt in prompt_results:
        if prompt not in prompts:
            continue
        #prompt = 'base_personal'
        prompt_path = os.path.join(target_directory, prompt)
        file_path = os.path.join(prompt_path, 'all_responses.json')
    
        with open(file_path) as json_file:
            data = json.load(json_file)

        new_data = [] #{doc_id, prediction, ground_truth}
        for i, v in enumerate(data):
            idd = v.get('doc_id')
            gt = v.get('ground_truth')
            ans = v.get('generated_response')
            class_seg = ans[:25]
            negative = prompts_and_answers.get(prompt)
            #print(type(negative))
            if negative in ans:
                pred = 0
            else:
                pred = 1
            new_data.append({'doc_id': idd, 'prediction': pred, 'ground_truth': gt})

        data = new_data
        data_df = new_get_join(data)
        clean_json = []
        for i, v in data_df.iterrows():
            clean = {
                'doc_id': v.doc_id,
                'prediction': v.prediction,
                'ground_truth': v.ground_truth,
                'model': mname,
                'prompt': prompt
            }
            clean_json.append(clean)

        main_results += clean_json
    df = pd.DataFrame(main_results)
    return df

def calculate_accuracy(group):
    correct_predictions = (group['prediction'] == group['ground_truth']).sum()
    total_predictions = len(group)
    accuracy = correct_predictions / total_predictions
    return accuracy

def calculate_balanced_accuracy(group):
    return balanced_accuracy_score(group['ground_truth'], group['prediction'])

def calculate_f1(group):
    return f1_score(group['ground_truth'], group['prediction'])

def calc_prec(group):
    return precision_score(group['ground_truth'], group['prediction'])

def calc_rec(group):
    return recall_score(group['ground_truth'], group['prediction'])

def tpr(group):
    tn, fp, fn, tp = confusion_matrix(group['ground_truth'], group['prediction']).ravel()
    tpr = tp / (tp+fn)
    return tpr

def tnr(group):
    tn, fp, fn, tp = confusion_matrix(group['ground_truth'], group['prediction']).ravel()
    tnr = tn / (tn+fp)
    return tnr

def calculate_f2(group):
    return fbeta_score(group['ground_truth'], group['prediction'], beta=2)

def auroc(group):
    return roc_auc_score(group['ground_truth'], group['prediction'])

def prompt_performance(df):
    #accuracy_df = results_df.groupby(['model', 'prompt']).apply(lambda x: (x['prediction'] == x['ground_truth']).mean()).reset_index(name='accuracy')
    # Group by model and prompt, then apply the calculation for each metric
    grouped = df.groupby(['model', 'prompt'])
    accuracy_df = grouped.apply(calculate_accuracy).reset_index(name='accuracy')
    balanced_accuracy_df = grouped.apply(calculate_balanced_accuracy).reset_index(name='balanced_accuracy')
    f1_score_df = grouped.apply(calculate_f1).reset_index(name='f1_score')
    prec_df = grouped.apply(calc_prec).reset_index(name='prec')
    rec_df = grouped.apply(calc_rec).reset_index(name='recall')
    tpr_df = grouped.apply(tpr).reset_index(name='tpr')
    tnr_df = grouped.apply(tnr).reset_index(name='tnr')
    f2_score_df = grouped.apply(calculate_f2).reset_index(name='f2_score')
    auroc_df = grouped.apply(auroc).reset_index(name='auroc')

    # Merge results into a single DataFrame - easy comparison
    performance_df = accuracy_df
    performance_df = pd.merge(performance_df, prec_df, on=['model', 'prompt'])
    performance_df = pd.merge(performance_df, tpr_df, on=['model', 'prompt'])
    performance_df = pd.merge(performance_df, tnr_df, on=['model', 'prompt'])
    performance_df = pd.merge(performance_df, f1_score_df, on=['model', 'prompt'])
    performance_df = pd.merge(performance_df, f2_score_df, on=['model', 'prompt'])
    performance_df = pd.merge(performance_df, balanced_accuracy_df, on=['model', 'prompt'])
    performance_df = pd.merge(performance_df, auroc_df, on=['model', 'prompt'])
    #performance_df = pd.merge(performance_df, rec_df, on=['model', 'prompt'])
    return performance_df


s = load_sara()
clean_unique_docs = no_reply_proc(s)
prompts = ['multi_category', 'text', 'pdc2', 'cg', 'textfew', 'pdcfew', 'cgfew']
x = get_results_json('mist-noreply')
#print(x)
prompt_performance_df = prompt_performance(x)
prompt_order = ['multi_category', 'text', 'pdc2', 'cg', 'textfew', 'pdcfew', 'cgfew']
prompt_performance_df['prompt'] = pd.Categorical(prompt_performance_df['prompt'], categories=prompt_order, ordered=True)
prompt_performance_df = prompt_performance_df.sort_values('prompt')
print(prompt_performance_df)


