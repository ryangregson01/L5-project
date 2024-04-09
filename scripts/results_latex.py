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
from nameless_preprocess import nameless_preproc
import spacy

def no_reply_proc(s, tokenizer='', c_size=2048):
    def clean_names(data, replaced=''):
        nlp = spacy.load("en_core_web_sm")
        anon_text = []
        for d in data.text:
            doc = nlp(d)
            anonymized_text = d
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    anonymized_text = anonymized_text.replace(ent.text, replaced)
            anon_text.append(anonymized_text)
        #data = data.reset_index()
        new_list = [{'doc_id':r.doc_id, 'text':anon_text[i], 'sensitivity':r.sensitivity} for i, r in data.iterrows()]
        return pd.DataFrame.from_dict(new_list)
    
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
    
    def remove_unnecessary(x):
        stop_words_extra = [' from ', ' e ', ' mail ', ' cc ', 'forwarded', ' by ', 'original ', ' message ', ' enron ', ' on ', ' pm ', ' am ']
        for w in stop_words_extra:
            x = re.sub(w, ' ', x)
        return x

    def main(s):
        #s = clean_names(s)
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
        #preproc_df['text'] = preproc_df['text'].apply(lambda x: remove_unnecessary(x))
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
                           'textqa': 'does not',
                           'pdcqa': 'does not',
                           'cgqa': 'does not',
                           'cgcot': 'non-personal',
                           'hop1': 'non-personal',
                           'detailsfew': 'non-personal'
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
            negative = 'non-personal'
            if negative in ans:
                pred = 0
            else:
                pred = 1
            new_data.append({'doc_id': idd, 'prediction': pred, 'ground_truth': gt})

        data = new_data
        data_df = new_get_join(data)
        clean_json = []
        for i, v in data_df.iterrows():
            if v.doc_id in X_train:
                #print(v.doc_id)
                #exit(0)
                continue
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
    return f1_score(group['ground_truth'], group['prediction'], average=average_type)

def calc_prec(group):
    return precision_score(group['ground_truth'], group['prediction'], average=average_type)

def calc_rec(group):
    return recall_score(group['ground_truth'], group['prediction'], average=average_type)

def tpr(group):
    tn, fp, fn, tp = confusion_matrix(group['ground_truth'], group['prediction']).ravel()
    tpr = tp / (tp+fn)
    return tpr

def tnr(group):
    tn, fp, fn, tp = confusion_matrix(group['ground_truth'], group['prediction']).ravel()
    tnr = tn / (tn+fp)
    return tnr

def calculate_f2(group):
    return fbeta_score(group['ground_truth'], group['prediction'], beta=2, average=average_type)

def auroc(group):
    return roc_auc_score(group['ground_truth'], group['prediction'])

def prompt_performance(df):
    #accuracy_df = results_df.groupby(['model', 'prompt']).apply(lambda x: (x['prediction'] == x['ground_truth']).mean()).reset_index(name='accuracy')
    # Group by model and prompt, then apply the calculation for each metric
    grouped = df.groupby(['model', 'prompt'])
    accuracy_df = grouped.apply(calculate_accuracy, include_groups=False).reset_index(name='Accuracy')
    balanced_accuracy_df = grouped.apply(calculate_balanced_accuracy, include_groups=False).reset_index(name='BAC')
    f1_score_df = grouped.apply(calculate_f1, include_groups=False).reset_index(name='$F_{1}$')
    prec_df = grouped.apply(calc_prec, include_groups=False).reset_index(name='Precision')
    rec_df = grouped.apply(calc_rec, include_groups=False).reset_index(name='Recall')
    tpr_df = grouped.apply(tpr, include_groups=False).reset_index(name='TPR')
    tnr_df = grouped.apply(tnr, include_groups=False).reset_index(name='TNR')
    f2_score_df = grouped.apply(calculate_f2, include_groups=False).reset_index(name='$F_{2}$')
    auroc_df = grouped.apply(auroc, include_groups=False).reset_index(name='auROC')

    # Merge results into a single DataFrame - easy comparison
    performance_df = accuracy_df
    performance_df = pd.merge(performance_df, prec_df, on=['model', 'prompt'])
    performance_df = pd.merge(performance_df, rec_df, on=['model', 'prompt'])
    performance_df = pd.merge(performance_df, tpr_df, on=['model', 'prompt'])
    performance_df = pd.merge(performance_df, tnr_df, on=['model', 'prompt'])
    performance_df = pd.merge(performance_df, f1_score_df, on=['model', 'prompt'])
    performance_df = pd.merge(performance_df, f2_score_df, on=['model', 'prompt'])
    performance_df = pd.merge(performance_df, balanced_accuracy_df, on=['model', 'prompt'])
    performance_df = pd.merge(performance_df, auroc_df, on=['model', 'prompt'])
    #performance_df = pd.merge(performance_df, rec_df, on=['model', 'prompt'])

    #df['DecimalCol'] = df['DecimalCol'].apply(lambda x: round(x, 2))
    return performance_df

def fix_name(mname):                
    if mname == 'mist-noreply' or mname=='mist7b-mist':
        return 'Mistral'
    elif mname == 'mixt-noreply' or mname=='mixt-4bit':
        return 'Mixtral'
    elif mname == 'l27b-noreply' or mname=='l27b-meta':
        return 'Llama 2'

def fix_prompts(p):
    symbol = {'base': 'Base', 
    'sens_cats': 'SensCat', 
    'all_cats': 'SensCat+NonSensCat', 
    'base_sens': 'Base+SensDesc', 
    'sens_cats_sens': 'SensCat+SensDesc', 
    'all_cats_sens': 'SensCat+NonSensCat+SensDesc', 
    'base_few': 'Base+FS', 
    'sens_cats_few': 'SensCat+FS', 
    'all_cats_few': 'SensCat+NonSensCat+FS', 
    'base_sens_few': 'Base+SensDesc+FS', 
    'sens_cats_sens_few': 'SensCat+SensDesc+FS', 
    'all_cats_sens_few': 'SensCat+NonSensCat+SensDesc+FS', 
    'all_cats_sens_hop1': 'SensCat+NonSensCat+SensDesc+CoT'}

    return symbol.get(p)

def round_df(df):
    for v in df.keys():
        if v == 'model' or v == 'prompt':
            if v == 'model':
                df[v] = df[v].apply(lambda x: fix_name(x))
            else:
                df[v] = df[v].apply(lambda x: fix_prompts(x))
            continue
        df[v] = df[v].apply(lambda x: round(x, 4))
    return df

s = load_sara()
clean_unique_docs = no_reply_proc(s)

from sklearn.model_selection import train_test_split
data = clean_unique_docs
X = data.doc_id.to_numpy()
y = data.sensitivity.to_numpy()
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.8, random_state=1)
X_train = [] # For full zero-shot

prompts = ['text', 'pdc2', 'cg', 'textfew', 'pdcfew', 'cgfew', 'hop1']
prompts = ['base', 'sens_cats', 'all_cats', 'base_sens', 'sens_cats_sens', 'all_cats_sens', 'base_few', 'sens_cats_few', 'all_cats_few', 'base_sens_few', 'sens_cats_sens_few', 'all_cats_sens_few', 'all_cats_sens_hop1']

prompts = ['base_few', 'sens_cats_few', 'all_cats_few', 'base_sens_few', 'sens_cats_sens_few', 'all_cats_sens_few']

model_name = ['mist-noreply', 'mixt-noreply', 'l27b-noreply', 'flanxl-noreply', 'mist-noreply-nameless']
model_name = model_name[2]
x = get_results_json(model_name)
average_type='binary'
prompt_performance_df = prompt_performance(x)
'''
df = pd.DataFrame()
model_names = ['mist-noreply', 'mixt-noreply'] #, 'l27b-noreply']
average_type='binary'
for model_name in model_names:
    x = get_results_json(model_name)
    prompt_performance_df = prompt_performance(x)
    if df.empty:
        df = prompt_performance_df
    else:
        df = pd.concat([df, prompt_performance_df], axis=0, ignore_index=True)
prompt_performance_df = df
model_order = ['mist-noreply', 'mixt-noreply']
prompt_performance_df['model'] = pd.Categorical(prompt_performance_df['model'], categories=model_order, ordered=True)
prompt_performance_df = prompt_performance_df.sort_values('model')
model_name = 'full'
'''
prompt_order = ['base', 'sens_cats', 'all_cats', 'base_sens', 'sens_cats_sens', 'all_cats_sens', 'base_few', 'sens_cats_few', 'all_cats_few', 'base_sens_few', 'sens_cats_sens_few', 'all_cats_sens_few', 'all_cats_sens_hop1']
prompt_performance_df['prompt'] = pd.Categorical(prompt_performance_df['prompt'], categories=prompt_order, ordered=True)
prompt_performance_df = prompt_performance_df.sort_values('prompt')
#print(prompt_performance_df)

rounded_df = round_df(prompt_performance_df)
print(rounded_df)
rounded_df = rounded_df.drop(columns=['Recall', 'auROC'])
rounded_df.to_csv(model_name+'_results.csv', index=False)
