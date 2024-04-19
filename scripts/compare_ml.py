import ir_datasets
import email
import pandas as pd
import re
import numpy as np
import json
import os
import sys
from dataset import load_sara
import gensim
from sklearn.model_selection import train_test_split
from ml import main
from results_latex import prompt_performance, round_df


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
    prompt_results = os.listdir(target_directory)
    main_results = []
    for prompt in prompt_results:
        if prompt not in prompts:
            continue
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


from statsmodels.stats.contingency_tables import mcnemar

def run_mcnemar(table):
    result = mcnemar(table, exact=True, correction=True)
    return result.statistic, result.pvalue

def mcnemar_table(control, change):
    control = control.sort_values(by='doc_id')
    change = change.sort_values(by='doc_id')
    model1_predictions = control.prediction.to_list()
    model2_predictions = change.prediction.to_list()

    model1_predictions = [0 if pred == truth else 1 for pred, truth in zip(model1_predictions, control.ground_truth.to_list())]
    model2_predictions = [0 if pred == truth else 1 for pred, truth in zip(model2_predictions, change.ground_truth.to_list())]

    contingency_table = [[0, 0], [0, 0]]
    for pred1, pred2 in zip(model1_predictions, model2_predictions):
        contingency_table[int(pred1)][int(pred2)] += 1

    return contingency_table

def mc_eval_util_both(results_df, p1, p2, m1, m2):
    orig = results_df[(results_df.model == m1) & (results_df.prompt==p1)]
    pure = results_df[(results_df.model == m2) & (results_df.prompt==p2)]
    table = mcnemar_table(orig, pure)
    return table

def mcnemar_eval(model_name, model_name2, prompt_name, prompt_name2, df):
    results_df = df[(df.model == model_name) | (df.model == model_name2)]
    overall_table = mc_eval_util_both(results_df, prompt_name, prompt_name2, model_name, model_name2)
    stat, p = run_mcnemar(overall_table)
    return {'prompt1':prompt_name, 'prompt2':prompt_name2, 'model': model_name, 'model2': model_name2, 'statistic':stat, 'p-value':('%.2E' % p), 'significant':(p < 0.05)}

def full_mcnemar_prompts_helper(df):
    comp_set1 = [('base', 'MF', 'mist-noreply', 'MF'), ('sens_cats_sens_few', 'MF', 'mist-noreply', 'MF'), ('base', 'Rand', 'mist-noreply', 'Rand'), ('sens_cats_sens_few', 'Rand', 'mist-noreply', 'Rand')]
    comp_set2 = [('base', 'LR', 'mist-noreply', 'LR'), ('sens_cats_sens_few', 'LR', 'mist-noreply', 'LR'), ('base', 'SVM', 'mist-noreply', 'SVM'), ('sens_cats_sens_few', 'SVM', 'mist-noreply', 'SVM')]
    comp_set = comp_set1+comp_set2
    stat_tests = []
    for v in comp_set:
        prompt_name = v[0]
        prompt_name2 = v[1]
        model_name = v[2]
        model_name2 = v[3]
        instance_stats = mcnemar_eval(model_name, model_name2, prompt_name, prompt_name2, df)
        stat_tests.append(instance_stats)
    stat_tests = pd.DataFrame.from_dict(stat_tests)
    print(stat_tests)
    return stat_tests

np.random.seed(1)
s = load_sara()
clean_unique_docs = no_reply_proc(s)
data = clean_unique_docs
X = data.doc_id.to_numpy()
y = data.sensitivity.to_numpy()
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.8, random_state=1)
#X_train = [] # For full zero-shot

prompts = ['base','sens_cats_sens_few']
df = pd.DataFrame()
model_names = ['mist-noreply']
for model_name in model_names:
    x = get_results_json(model_name)

ml_df = main()
df = pd.concat([ml_df, x], axis=0, ignore_index=True)

full_mcnemar_prompts_helper(df)

y = prompt_performance(df)
rounded_df = round_df(y)
rounded_df = rounded_df.drop(columns=['Recall', 'auROC'])
print(rounded_df)
rounded_df.to_csv('ml'+'_results.csv', index=False)