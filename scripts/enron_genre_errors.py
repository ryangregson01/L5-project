import json
import pandas as pd
import os
import numpy as np
import pandas as pd
import re
import email
import gensim
import string
import sys
sys.path.append("/")
from scripts.dataset import load_sara
from scripts.preprocess_sara import full_preproc
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import spacy
import glob

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


def new_get_join(data):
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

def clean_names(data):
    nlp = spacy.load("en_core_web_sm")
    anon_text = []
    for d in data.text:
        doc = nlp(d)
        anonymized_text = d
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                anonymized_text = anonymized_text.replace(ent.text, "")
        anon_text.append(anonymized_text)
    new_list = [{'doc_id':r.doc_id, 'text':anon_text[i], 'sensitivity':r.sensitivity} for i, r in data.iterrows()]
    return pd.DataFrame.from_dict(new_list)

def get_results_json(mname, clean=True):
    current_directory = os.getcwd()
    target_directory = current_directory + f'/scripts/results/model_results/{mname}/'
    #print("Path to results", target_directory)
    prompt_results = os.listdir(target_directory)
    main_results = []
    ps = ['gdpr_qa', 'context1_qa', 'context2_qa', 'workemail_qa', 'context1_fewshot_qa', 'context1_class']
    ps = ['text', 'pdc', 'cg', 'textqa', 'pdcqa', 'cgqa']
    ps = ['textfew', 'pdcfew', 'cgfew']
    ps = ['pdcfewsim', 'pdc2']
    ps = ['pdcfew']

    prompt = ps[0] #'text2'
    prompt_path = os.path.join(target_directory, prompt)
    file_path = os.path.join(prompt_path, 'all_responses.json')
    #print(file_path)
    
    with open(file_path) as json_file:
        data = json.load(json_file)

    new_data = [] #{doc_id, prediction, ground_truth}
    for i, v in enumerate(data):
        idd = v.get('doc_id')
        gt = v.get('ground_truth')
        ans = v.get('generated_response')
        #ans = 'The text does contain sensitive'
        class_seg = ans[:25]
        negative = 0
        #if 'non-personal' in ans:
        #    pred = 0
        #else:
        #    pred = 1

        if 'purely' in ans or 'but' in ans:
            pred = 1
        else:
            pred = 0



        new_data.append({'doc_id': idd, 'prediction': pred, 'ground_truth': gt})
    
    data = new_data
    #print(data)
    right_sum = [0,0,0,0,0,0,0,0]
    total_sum = [0,0,0,0,0,0,0,0]
    data_df = new_get_join(data)
    for i, v in data_df.iterrows():
        idd = v.doc_id
        classs = int(mapp.get(idd))
        total_sum[(classs-1)] += 1
        if v.prediction == v.ground_truth:
            right_sum[(classs-1)] += 1


        #break
    print(total_sum)
    print(right_sum)
    exit(0)
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

print(1)
s = load_sara()
dddd = full_preproc(s)
email_files = [f.replace('.cats', '') for f in glob.glob('./*/*/*.cats')]
mapp = {}
for v in email_files:
    v = v.split('/')
    mapp[v[-1]] = v[-2]
    #break
#print(mapp)

x = get_results_json('mist-noreply')
print(x)
