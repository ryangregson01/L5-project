import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import gc
import time
import numpy as np
from prompts import *


from dataset import load_sara
from few import get_key_to_sims, get_sims, get_sim_text

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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encodeds = tokenizer(document, return_tensors="pt", padding=True)
    #print((encodeds.input_ids.size()))
    model_inputs = encodeds.to(device)
    generated_ids = model.generate(inputs=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pad_token_id=tokenizer.pad_token_id, max_new_tokens=150) #150)
    decoded = tokenizer.batch_decode(generated_ids)
    del model_inputs
    torch.cuda.empty_cache()
    gc.collect()
    #print(decoded[0])
    return decoded

'''
def llm_inference(document, prompt, model, tokenizer):
    #Tokenizes input prompt with document using a chat template, generates text, decodes text.
    messages = [
    #{"role": "system", "content": "You are identifying documents containing personal sensitive information."},
    {"role": "user", "content": prompt(document)},
    #{"role": "assistant", "content": "This text is {'mask'}"}
    ]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    device = 'cuda'
    model_inputs = encodeds.to(device)
    arr_like = torch.ones_like(model_inputs)
    attention_mask = arr_like.to(device)
    generated_ids = model.generate(inputs=model_inputs, attention_mask=attention_mask, max_new_tokens=10, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    #print(decoded)
    return decoded[0]
'''

def display_gen_text(output, e):
    '''Segments response to only new generated text.'''
    end_template = output.split(e)
    return end_template[-1]


def prompt_to_reply(d, p, m, t, e, device):
    '''Gets response from model.'''
    response = llm_inference(d, p, m, t, device)
    gen_text = []
    for r in response:
        gen = display_gen_text(r, e)
        gen_text.append(gen)
    return gen_text


def post_process_classification(classification):
    '''String matching on model response'''
    match_string = classification.lower()
    if 'does contain' in match_string:
        return 1
    elif 'does not' in match_string:
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

    # proc is dataset, get key_to_sims for fewshot
    #s = load_sara()
    #proc = dataset
    #size = 5
    #key_to_sims = get_key_to_sims(size)

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

        # Input text is too large for model
        if len(sample_text) > 10000:
            fpr[sample_id] = "TOO LARGE"
            mr[sample_id] = "TOO LARGE"
            continue

        '''
        document = sample_text
        ds = proc[proc.text == document]
        idd = ds.doc_id.iloc[0]
        if '_' in idd:
            idd = idd[:idd.find('_')]
        l = key_to_sims.get(idd)
        len_doc = len(document)
        remaining_text_space = 9500 - len_doc
        each_example = remaining_text_space / 2
        few_sens_ex = ''
        few_nonsens_ex = ''

        for sens in l[1]:
            senstext = get_sim_text(s, sens, proc)
            if len(senstext) <= each_example:
                few_sens_ex = senstext
                break

        for nonsens in l[0]:
            nonsenstext = get_sim_text(s, nonsens, proc)
            if len(nonsenstext) <= each_example:
                few_nonsens_ex = nonsenstext
                break

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
        #mr[sample_id] = classification
        for i, k in enumerate(batch_ids):
            mr[k] = classification[i]

            pred = post_process_classification(classification[i])
            if pred == None:
                # Generated classification could not be identified using processing.
                fpr[k] = classification[i]
                continue

            truths.append(ground_truth)
            preds.append(pred)

        batch_ids = []

        clear_memory()

    return preds, truths, mr, fpr


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
