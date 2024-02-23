import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import gc
import time
import numpy as np
from prompts import *


def llm_inference(document, prompt, model, tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encodeds = tokenizer(prompt(document), return_tensors="pt")
    device = 'cuda'
    model_inputs = encodeds.to(device)
    generated_ids = model.generate(inputs=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pad_token_id=tokenizer.pad_token_id, max_new_tokens=10, do_sample=False)
    decoded = tokenizer.batch_decode(generated_ids)
    del model_inputs
    torch.cuda.empty_cache()
    gc.collect()
    return decoded[0]

'''
def llm_inference(document, prompt, model, tokenizer):
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
  end_template = output.find(e)
  return output[end_template:]


def prompt_to_reply(d, p, m, t, e):
  response = llm_inference(d, p, m, t)
  return display_gen_text(response, e)


# String matching on model response
def post_process_classification(classification, ground_truth):
    if "does contain" in classification.lower():
        if ground_truth == 1:
            return 'TP', 1
        else:
            return 'FP', 1
    
    elif "does not" in classification.lower():
        if ground_truth == 0:
            return 'TN', 0
        else:
            return 'FN', 0

    else:
        # Further processing required
        return classification, None
        further_processing_required[sample[1].doc_id] = classification


def clear_memory():
    # Prevents cuda out of memory
    torch.cuda.empty_cache()
    gc.collect()


# Dataset - dataframe, prompt_strategy - prompt function name, model - LLM
def llm_experiment(dataset, prompt_strategy, model, tokenizer, end_prompt=None):
    predictions = {
        'TP' : 0, # Sensitive
        'FP' : 0, # Non-sensitive document classified as sensitive
        'TN' : 0, # Non-sensitive
        'FN' : 0,
    }
    # Model output is not an expected sensitivity attribute
    further_processing_required = {}
    # All model output
    model_responses = {}

    scikit_true = []
    scikit_pred = []

    for sample in dataset.iterrows():
        sample_text = sample[1].text
        ground_truth = sample[1].sensitivity

        # To replace with appropriate pre-processing
        if len(sample_text) > 10000:
            continue

        classification = prompt_to_reply(sample_text, prompt_strategy, model, tokenizer, end_prompt)
        model_responses[sample[1].doc_id] = classification

        quadrant, pred = post_process_classification(classification, ground_truth)
        if pred == None:
            further_processing_required[sample[1].doc_id] = quadrant
            continue

        predictions[quadrant] = predictions.get(quadrant) + 1
        scikit_true.append(ground_truth)
        scikit_pred.append(pred)

        clear_memory()

    return predictions, further_processing_required, model_responses, scikit_true, scikit_pred


def post_process_split_docs(mr, fpr, pre, df):
    clean_doc_id = {}
    ground_truths = []
    ite = -1
    for s in mr.keys(): #samp.doc_id():
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
    return values_array, ground_truths
