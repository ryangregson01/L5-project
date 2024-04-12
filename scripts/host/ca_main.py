#import ir_datasets
import pandas as pd
import numpy as np
import re
import email
import gensim
import string
import torch
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from ca_prompts import *
import gc
import time
import json
import os
import sys
#from few import get_key_to_sims, get_sims, get_sim_text, new_get_sims
#from config import my_cache

model_map = {'test-mist': ['get_model', 'mistralai/Mistral-7B-Instruct-v0.2', 'main'],
            'mist-noreply': ['get_model', 'mistralai/Mistral-7B-Instruct-v0.2', 'main'],
            'l27b-noreply': ['get_l2', 'meta-llama/Llama-2-7b-chat-hf', 'main'],
            'mixt-noreply': ['get_model', 'TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ', 'main'], #'gptq-4bit-32g-actorder_True'], #'main'],
            }
l2_token = ""

# DATASET
'''
def get_sara():
    return ir_datasets.load('sara')

def dataset_to_df(dataset):
    doc_ids = []
    doc_text = []
    doc_sens = []
    for doc in dataset.docs_iter():
        doc_ids.append(doc.doc_id)
        doc_text.append(doc.text)
        doc_sens.append(doc.sensitivity)

    sara_dict = {'doc_id':doc_ids, 'text':doc_text, 'sensitivity':doc_sens}
    df = pd.DataFrame.from_dict(sara_dict)
    return df

def load_sara():
    sara_dataset = get_sara()
    sara_df = dataset_to_df(sara_dataset)
    return sara_df
'''
def get_flan(v, r, d):
    model_name = v
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=d, revision=r, trust_remote_code=False)
    return tokenizer, model


# MODELS LOAD
def get_l2(version, revision, device):
    access_token = l2_token
    model_name = version
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=access_token, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, token=access_token, revision=revision)
    return tokenizer, model

def get_l2_bits(version, revision, device):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    access_token = l2_token
    model_name = version
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=access_token, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, quantization_config=bnb_config , token=access_token, revision=revision)
    return tokenizer, model

def get_model(v, r, d):
    model_name = v
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=d, revision=r, trust_remote_code=False) #, cache_dir=my_cache)
    return tokenizer, model

def get_model_bnb(v, r, d):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_name = v
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=d, quantization_config=bnb_config, revision=r)
    return tokenizer, model

def get_model_version(m, v, r, d):
    models_dict = {'get_l2' : get_l2, 'get_model': get_model, 'get_l2_bits':get_l2_bits, 'get_model_bnb':get_model_bnb}
    return models_dict.get(m)(v, r, d)


# PREPROCESS
def full_preproc(s, tokenizer, c_size=2048):

    def preprocess(e):
        message = email.message_from_string(e)
        clean = message.get_payload()
        clean = re.sub('\S*@\S*\s?', '', clean)
        clean = re.sub('\s+', ' ', clean)
        clean = re.sub("\'", "", clean)
        clean = gensim.utils.simple_preprocess(str(clean), deacc=True, min_len=1, max_len=100) 
        #clean = clean.lower()
        #clean = clean.translate(str.maketrans('','', string.punctuation))
        #clean = clean.translate(str.maketrans('','', "-_?"))
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

    def get_replies(df):
        place = []
        for i, tex in enumerate(df.text):
            words = tex.split()
            for j, word in enumerate(words):

                if 'forwarded' == word:
                    if words[j+1] == 'by':
                        place.append((i, j))
                        continue

                if 'original' == word:
                    if words[j+1] == 'message':
                        place.append((i, j))
                        continue

        return place

    def chunk(text, tokenizer, c_size):
        new_chunks = []
        tokens= tokenizer(text, return_tensors="pt")
        total_length = len(tokens.input_ids[0])
        avg_chunks = np.ceil(total_length / c_size)
        for i in range(int(avg_chunks)):
            chunk = tokens.input_ids[0][(i*c_size):((i+1)*c_size)]
            chunk = tokenizer.decode(chunk, skip_special_tokens=True)
            new_chunks.append(chunk)

        return new_chunks
            
    def chunk_large(df, place, tokenizer, c_size):
        place_docs = [dno[0] for dno in place]
        new_docs = []
        existing_texts = []
        for i, s in enumerate(df.iterrows()):
            ids = s[1].doc_id
            sens = s[1].sensitivity
            te = s[1].text

            if i not in place_docs:
                
                new_chunks = chunk(te, tokenizer, c_size)
                if len(new_chunks) == 1:
                    new_docs.append({'doc_id':str(ids), 'text':te, 'sensitivity':sens})
                    continue

                cut = 0
                for c in new_chunks:
                    new_docs.append({'doc_id':str(ids)+'_'+str(cut), 'text':c, 'sensitivity':sens})
                    cut += 1
                continue

            words = te.split()
            cut_pos_init = 0
            cut = 0
            for pair in place:
                if pair[0] == i:
                    cut_pos = pair[1]
                    seg = words[cut_pos_init:cut_pos]
                    cut_pos_init = cut_pos
                    text_join = ' '.join(seg)
                    
                    new_chunks = chunk(text_join, tokenizer, c_size)
                    if len(new_chunks) == 1:
                        x = {'doc_id':ids+'_'+str(cut), 'text':text_join, 'sensitivity':sens}
                        cut += 1
                        new_docs.append(x)

                    else:
                        for c in new_chunks:
                            new_docs.append({'doc_id':ids+'_'+str(cut), 'text':c, 'sensitivity':sens})
                            cut += 1

            seg = words[cut_pos_init:]
            text_join = ' '.join(seg)
            
            new_chunks = chunk(text_join, tokenizer, c_size)
            if len(new_chunks) == 1:
                x = {'doc_id':ids+'_'+str(cut), 'text':text_join, 'sensitivity':sens}
                cut += 1
                new_docs.append(x)
            else:
                for c in new_chunks:
                    new_docs.append({'doc_id':ids+'_'+str(cut), 'text':c, 'sensitivity':sens})
                    cut += 1
            
        return new_docs

    def main(s, tokenizer):
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
        #places = get_replies(preproc_df)
        if type(tokenizer) == str:
            tokenizer = model_map.get(tokenizer)[1]
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        places = []
        new_docs = chunk_large(preproc_df, places, tokenizer, c_size)
        new_docs = pd.DataFrame.from_dict(new_docs)
        return new_docs

    return main(s, tokenizer)

## MODEL INFERENCE
def llm_inference(document, prompt, model, tokenizer, device):
    device = device
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
            max_new_tokens=10, # 150
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

    batch = []
    batch_ids = []
    cur_bs = 32
    count = 0
    for sample in dataset.iterrows():
        if (count % 100) == 0:
            print('COUNT: ', count)
        count += 1

        sample_id = sample[1].doc_id
        sample_text = sample[1].text
        ground_truth = sample[1].sensitivity

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

        batch_ids = []

        clear_memory()

    return mr #, full_prompt


# WRITE OUT
def all_responses_json(model_responses, model_name, prompt_name): #, full_p):
    results = []
    ite = -1
    for val in model_responses.keys():
        result = {
            'model': model_name,
            'prompt': prompt_name,
            'doc_id': val,
            'generated_response': model_responses[val]
            #'full_response': full_p[val]
        }
        results.append(result)

    return results


def pipeline(mname, d, prompt_no=0, n=None):
    #sara_df = load_sara()
    sara_df = pd.read_csv('sara.csv')

    model_list = model_map.get(mname)
    m = model_list[0]
    v = model_list[1]
    r = model_list[2]
    if d == 'cpu':
        tokenizer, model = get_model_version(m, v, r, d)
    else:
        tokenizer, model = get_model_version(m, v, r, 'auto')
    
    if n == None:
        processed_sara_df = full_preproc(sara_df, tokenizer)
    else:
        n = int(n)
        samp = sara_df.sample(n=n, random_state=1)
        processed_sara_df = full_preproc(samp, tokenizer)

    #print(processed_sara_df)
    prompt_list = ['base', 'sens_cats', 'all_cats', 'base_few', 'sens_cats_few', 'all_cats_few', 'base_sens', 'sens_cats_sens', 'all_cats_sens', 'base_sens_few', 'sens_cats_sens_few', 'all_cats_sens_few', 'all_cats_sens_hop1', 'all_cats_sens_hop2', 'all_cats_sens_hop3']
    prompt_no = int(prompt_no)
    prompt_name = prompt_list[prompt_no]
    print('Prompt='+prompt_name)
    prompt_str = mname + '_' + prompt_name + '_results'
    prompt = get_prompt_matrix(prompt_name)
    end_prompt = '[/INST]'
    model_responses = llm_experiment(processed_sara_df, prompt, model, tokenizer, d, end_prompt)
    #print(model_responses)
    results = all_responses_json(model_responses, mname, prompt_name)
    print('Writing to', prompt_str)
    with open('/nfs/'+prompt_str+'.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Done")


name = sys.argv[1]
d = sys.argv[2]
pno = sys.argv[3]

#pipeline(mname='test-mist', d='cpu', prompt_no="0", n="1")
#pipeline(mname=name, d=d, prompt_no=pno, n=n)
pipeline(mname=name, d=d, prompt_no=pno)


