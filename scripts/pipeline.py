from dataset import load_sara
from preprocess_sara import proccutit, full_preproc
from models import get_model_version
from model import llm_experiment, post_process_split_docs
import time
import numpy as np
import os
import json
from few import get_key_to_sims
#from prompts_matrix import get_prompt_matrix
from final_prompts import get_prompt_matrix
import spacy
import pandas as pd

from nameless_preprocess import nameless_preproc


def all_responses_json(model_responses, further_processing_required, preds_list, truths_list, model_name, prompt_name, sara_df): #, full_p):
    results = []
    ite = -1
    for val in model_responses.keys():
        '''
        if val in further_processing_required.keys():
            prediction = None
            s = val
            if '_' in s:
                s = s[:val.find('_')]
            # Must convert numpy int64 to integer, otherwise JSON cannot serialize
            ground_truth = int((sara_df[sara_df.doc_id == s].sensitivity).iloc[0])
        else:
            ite += 1
            prediction = preds_list[ite]
            ground_truth = truths_list[ite]
        '''
        result = {
            'model': model_name,
            'prompt': prompt_name,
            'doc_id': val,
            'generated_response': model_responses[val]
            #'prediction': prediction,
            #'ground_truth': ground_truth
            #'full_response': full_p[val]
        }
        results.append(result)

    return results

def clean_responses_json(doc_keys, preds, truths, model_responses, model_name, prompt_name):
    results = []
    for i, val in enumerate(doc_keys):        
        result = {
            'model': model_name,
            'prompt': prompt_name,
            'doc_id': val,
            #'generated_response': model_responses[val],
            'prediction': preds[i],
            'ground_truth': truths[i]
        }
        results.append(result)
    return results

def write_responses_json(results, filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
    except:
        data = []
   
    results = data + results
    with open(filename, 'w') as file:
        json.dump(results, file, indent=4)


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
    new_list = [{'doc_id':r.doc_id, 'text':anon_text[i], 'sensitivity':r.sensitivity} for i, r in data.iterrows()]
    return pd.DataFrame.from_dict(new_list)

def run_pipeline(model_name, m, v, r, d, prompts, end_prompt, n=None):
    """
    Runs full pipeline: downloading and preprocessing dataset, running experiment, 
    and writing out predictions from the model.
    
    Parameters:
    model_name (string)
    m (string): Function to load model (get_model or get_l2)
    v (string): Model path to download from HuggingFace.
    r (string): Revision of branch to download from HuggingFace (usually main).
    d (string): Device (usually auto).
    prompts (list of strings): Prompts used in experiment.
    end_prompt (string): Where generated text should start for processing model response.
    n (optional integer): To create a fixed sample of the dataset.
    """

    sara_df = load_sara()
    tokenizer, model = get_model_version(m, v, r, d)
    #sara_df = clean_names(sara_df)

    if n == None:
        processed_sara_df = full_preproc(sara_df, tokenizer)
    else:
        samp = sara_df.sample(n=n, random_state=1)
        processed_sara_df = full_preproc(samp, tokenizer)

    #key_to_sims = get_key_to_sims()
    
    #tokenizer, model = get_model_version(m, v, r, d)

    for prompt in prompts:
        prompt_name = prompt
        print('Using', prompt_name)
        prompt_str = 'results/model_results/' + model_name + '/' + prompt_name + '/'
        prompt = get_prompt_matrix(prompt)
        start = time.time()
        #preds_list, truths_list, model_responses, further_processing_required = llm_experiment(processed_sara_df, prompt, model, tokenizer, d, end_prompt) #key_to_sims, end_prompt)
        model_responses = llm_experiment(processed_sara_df, prompt, model, tokenizer, d, end_prompt) #key_to_sims, end_prompt)
        end = time.time()
        duration = end-start
        
        #new_preds, new_truths, doc_keys = post_process_split_docs(model_responses, further_processing_required, preds_list, sara_df)
        #truth_labs = np.array(new_truths)
        #preds = np.array(new_preds)
        if not os.path.exists(prompt_str):
            os.makedirs(prompt_str)
        # Save all predictions, relevant ground truths and model responses
        #np.savetxt(prompt_str+'truth_labs.txt', truth_labs)
        #np.savetxt(prompt_str+'preds.txt', preds)
        f = open(prompt_str+"duration.txt", "w")
        f.write(str(duration))
        f.close()

        #results = all_responses_json(model_responses, '', '', '', model_name, prompt_name, sara_df, full_p)
        results = all_responses_json(model_responses, '', '', '', model_name, prompt_name, sara_df)
        with open(prompt_str+'all_responses.json', 'w') as f:
            json.dump(results, f, indent=2)
        write_responses_json(results, 'results/all_model_responses.json')    
        #results = clean_responses_json(doc_keys, preds.tolist(), truth_labs.tolist(), model_responses, model_name, prompt_name)
        #write_responses_json(results, 'results/clean_model_responses.json')
