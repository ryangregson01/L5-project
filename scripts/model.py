import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import gc
import time
import numpy as np
from prompts import *


def llm_inference(document, prompt, model, tokenizer):
    '''Tokenizes input prompt with document, generates text, decodes text.'''
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encodeds = tokenizer(prompt(document), return_tensors="pt")
    device = 'cuda'
    model_inputs = encodeds.to(device)
    generated_ids = model.generate(inputs=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pad_token_id=tokenizer.pad_token_id, max_new_tokens=10)
    decoded = tokenizer.batch_decode(generated_ids)
    del model_inputs
    torch.cuda.empty_cache()
    gc.collect()
    return decoded[0]

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
    end_template = output.find(e)
    return output[end_template:]


def prompt_to_reply(d, p, m, t, e):
    '''Gets response from model.'''
    response = llm_inference(d, p, m, t)
    return display_gen_text(response, e)


def post_process_classification(classification):
    '''String matching on model response'''
    match_string = classification.lower()
    if 'sensitive' in match_string and 'non-sensitive' not in match_string:
        return 1
    elif 'non-sensitive' in match_string:
        return 0
    else:
        # Further processing required
        return None


def clear_memory():
    # Prevents cuda out of memory
    torch.cuda.empty_cache()
    gc.collect()


def llm_experiment(dataset, prompt_strategy, model, tokenizer, end_prompt=None):
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

    for sample in dataset.iterrows():
        sample_id = sample[1].doc_id
        sample_text = sample[1].text
        ground_truth = sample[1].sensitivity

        # Input text is too large for model
        if len(sample_text) > 10000:
            fpr[sample_id] = "TOO LARGE"
            continue

        classification = prompt_to_reply(sample_text, prompt_strategy, model, tokenizer, end_prompt)
        mr[sample_id] = classification

        pred = post_process_classification(classification)
        if pred == None:
            # Generated classification could not be identified using processing.
            fpr[sample_id] = classification
            continue

        truths.append(ground_truth)
        preds.append(pred)

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
