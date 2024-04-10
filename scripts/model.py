import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import gc
import time
import numpy as np
from prompts import *
from prompts_matrix import hop2, hop3, pdc, fewshotsimone
from final_prompts import all_cats_sens_hop1, all_cats_sens_hop2, all_cats_sens_hop3


from dataset import load_sara
from preprocess_sara import full_preproc
from few import get_key_to_sims, get_sims, get_sim_text, new_get_sims

import json
import os
def read_cots():
    path = os.getcwd() + '/results/model_results/mist7b-mist/itspersonalverbose/'
    file_path = path + 'resp.json'
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def llm_inference(document, prompt, model, tokenizer, device):
    device = 'cuda'
    '''Tokenizes input prompt with document, generates text, decodes text.'''
    #tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token #"<pad>" #'[PAD]' #tokenizer.eos_token
    encodeds = tokenizer(document, return_tensors="pt", padding=True)
    model_inputs = encodeds.to(device)
    with torch.no_grad():
        generated_ids = model.generate(inputs=model_inputs.input_ids, 
            attention_mask=model_inputs.attention_mask, 
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            max_new_tokens=150, # 150
        )
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    del model_inputs
    torch.cuda.empty_cache()
    gc.collect()
    return decoded

def cot_helper(model, tokenizer, model_inputs, tokens=10):
    with torch.no_grad():
        generated_ids = model.generate(inputs=model_inputs.input_ids, 
            attention_mask=model_inputs.attention_mask, 
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            max_new_tokens=tokens, # 150
        )
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    del model_inputs
    torch.cuda.empty_cache()
    gc.collect()
    return decoded

def llm_inference_cot(document, prompt, model, tokenizer, device, plain_message):
    device = 'cuda'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encodeds = tokenizer(document, return_tensors="pt", padding=True)
    model_inputs = encodeds.to(device)
    decode_text = cot_helper(model, tokenizer, model_inputs, 60)[0]
    end_template = decode_text.split('Answer:')[-1]
    hop_document = all_cats_sens_hop2(plain_message, end_template)
    encodeds = tokenizer(hop_document, return_tensors="pt", padding=True)
    model_inputs = encodeds.to(device)
    decode_text = cot_helper(model, tokenizer, model_inputs, 60)[0]
    end_template2 = decode_text.split('Answer:')[-1]
    hop_document = all_cats_sens_hop3(plain_message, end_template+end_template2)
    encodeds = tokenizer(hop_document, return_tensors="pt", padding=True)
    model_inputs = encodeds.to(device)
    decoded = cot_helper(model, tokenizer, model_inputs)
    return decoded

def display_gen_text(output, e):
    '''Segments response to only new generated text.'''
    end_template = output.split(e)
    return end_template[-1]


def prompt_to_reply(d, p, m, t, e, device, cot_doc=''):
    '''Gets response from model.'''
    if cot_doc != '':
        response = llm_inference_cot(d, p, m, t, device, cot_doc)
    else:
        response = llm_inference(d, p, m, t, device)
    gen_text = []
    for r in response:
        gen = display_gen_text(r, e)
        gen_text.append(gen) #(gen, response[0]))
    return gen_text


def post_process_classification(classification):
    '''String matching on model response'''
    match_string = classification.lower()
    match_string = match_string[:50]
    #print(match_string)
    if 'does contain' in match_string or ('personal' in match_string and 'non-personal' not in match_string):
        return 1
    elif 'does not' in match_string or ('non-personal' in match_string):
        return 0
    else:
        # Further processing required
        return None


def clear_memory():
    # Prevents cuda out of memory
    torch.cuda.empty_cache()
    gc.collect()


def llm_experiment(dataset, prompt_strategy, model, tokenizer, device, end_prompt=None):
    """
    Run main experiment.
    
    Parameters:
    dataset (pandas dataframe): Dataframe with columns doc_id, text, sensitivity.
    prompt_strategy (function): Prompt template.
    model: generative LLM
    tokenizer
    end_prompt (string): 
    
    Returns:
    preds (array of integers): Predictions from post-processing model classifications.
    truths (array of integers): Ground truths.
    mr (model responses) (dictionary): Generated text for each input.
    fpr (further processing required) (dictionary): Unclassified text.
    """

    predictions = {
        'TP' : 0, # Sensitive
        'FP' : 0, # Non-sensitive document classified as sensitive
        'TN' : 0, # Non-sensitive
        'FN' : 0,
    }
    # Model output is not an expected sensitivity attribute
    fpr = {}
    # All model output
    mr = {}
    truths = []
    preds = []
    full_prompt = {}

    # proc is the sampled dataset, get key_to_sims for fewshot
    #full_proc = full_preproc(load_sara(), tokenizer)
    #proc = dataset
    #key_to_sims = get_key_to_sims(full_proc, proc, tokenizer)

    ds = dataset.sort_values(by=["text"],key=lambda x:x.str.len())
    dataset = ds

    #modelcots = read_cots()

    batch = []
    batch_ids = []
    cur_bs = 32
    count = 0
    for sample in dataset.iterrows():
        if (count % 100) == 0:
            print('\nCOUNT: ', count)
        count += 1

        sample_id = sample[1].doc_id
        sample_text = sample[1].text
        ground_truth = sample[1].sensitivity

        '''
        # Input text is too large for model
        #if len(sample_text) > 10000:
        #    fpr[sample_id] = "TOO LARGE"
        #    mr[sample_id] = "TOO LARGE"
        #    continue
        '''
        '''
        document = sample_text
        #ds = proc[proc.text == document]
        #idd = ds.doc_id.iloc[0]
        #if '_' in idd:
        #    idd = idd[:idd.find('_')]
        idd = sample_id
        l = key_to_sims.get(idd)
        len_doc = len(document)
        shot1, label1, shot2, label2 = new_get_sims(l, full_proc, len_doc)
        if label1 == 0:
            label1 = 'non-personal'
        elif label1 == 1:
            label1 = 'personal'
        if label2 == 0:
            label2 = 'non-personal'
        elif label2 == 1:
            label2 = 'personal'

        if label1 == -1:
            prompt_input = pdc(sample_text)
        elif label2 == -1:
            prompt_input = fewshotsimone(sample_text, shot1, label1)
        else:
            prompt_input = prompt_strategy(sample_text, shot1, label1, shot2, label2)
        '''
        '''
        k = sample_id
        thought = modelcots.get(k, 'No thought response.')
        #print(thought)
        if len(sample_text) + len(thought) > 10000:
            mr[sample_id] = "TOO LARGE with thought."
            continue
        
        '''

        prompt_input = prompt_strategy(sample_text) #, thought) #, few_sens_ex, few_nonsens_ex)
        batch.append(prompt_input)
        batch_ids.append(sample_id)
        if len(batch) == cur_bs or (count > 0):
            sample_text = batch
            batch = []
        else:
            continue

        classification = prompt_to_reply(sample_text, prompt_strategy, model, tokenizer, end_prompt, device)
        #classification = prompt_to_reply(sample_text, prompt_strategy, model, tokenizer, end_prompt, device, cot_doc=sample[1].text)
        #mr[sample_id] = classification
        for i, k in enumerate(batch_ids):
            #mr[k], full_prompt[k] = classification[i]
            mr[k] = classification[i]

            '''
            pred = post_process_classification(classification[i][0])
            if pred == None:
                # Generated classification could not be identified using processing.
                fpr[k] = classification[i]
                continue

            truths.append(ground_truth)
            preds.append(pred)
            '''
        batch_ids = []

        clear_memory()

    return mr #, full_prompt #preds, truths, mr, fpr #, full_prompt


def post_process_split_docs(mr, fpr, pre, df):
    clean_doc_id = {}
    ground_truths = []
    ite = -1
    for s in mr.keys():
        if s in fpr.keys():
            continue

        if '_' in s:
            s = s[:s.find('_')]

        val = clean_doc_id.get(s, -1)
        ite += 1

        if val == -1:

            clean_doc_id[s] = pre[ite]
            ground_truths.append((df[df.doc_id == s].sensitivity).iloc[0])
            continue

        if (val == pre[ite] or val == 1):
            continue
        
        clean_doc_id[s] = pre[ite]

    values_array = np.array(list(clean_doc_id.values()))
    clean_doc_keys = np.array(list(clean_doc_id.keys()))
    return values_array, ground_truths, clean_doc_keys
