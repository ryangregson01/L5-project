import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import gc
import time
import numpy as np
from prompts import *


def llm_inference(document, prompt, model, tokenizer):
  inputs = tokenizer(prompt(document), return_tensors='pt')
  generation_config = GenerationConfig(
      # Unable to set temperature to 0 - https://github.com/facebookresearch/llama/issues/687 - use do_sample=False for greedy decoding
      do_sample=False,
      max_new_tokens=10,
      max_time=180
  )
  output = model.generate(inputs=inputs.input_ids.cuda(), attention_mask=inputs.attention_mask.cuda(), generation_config=generation_config)
  response = tokenizer.decode(output[0], skip_special_tokens=True)
  return response
'''
def llm_inference(document, prompt, model, tokenizer):
  messages = [
    {"role": "user", "content": prompt(document)}
  ]
  encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
  inputs = encodeds.to('cuda')
  arr_like = torch.ones_like(inputs)
  attention_mask = arr_like.to('cuda')
  generated_ids = model.generate(inputs=inputs, attention_mask=attention_mask, max_new_tokens=50, do_sample=False)
  decoded = tokenizer.batch_decode(generated_ids)
  return decoded[0]

def llm_inference(document, prompt, model, tokenizer):
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  pad_token_id=tokenizer.pad_token_id
  inputs = tokenizer(prompt(document), return_tensors='pt')
  generation_config = GenerationConfig(
    # Unable to set temperature to 0 - https://github.com/facebookresearch/llama/issues/687 - use do_sample=False for greedy decoding
    do_sample=False,
    max_new_tokens=20,
  )
  output = model.generate(inputs=inputs.input_ids.cuda(), attention_mask=inputs.attention_mask.cuda(), pad_token_id=tokenizer.pad_token_id, generation_config=generation_config)
  return tokenizer.decode(output[0], skip_special_tokens=True)
'''

def display_gen_text(output):
  end_template = output.find('\nAnswer:')
  return output[end_template:]


def prompt_to_reply(d, p, m, t):
  response = llm_inference(d, p, m, t)
  return display_gen_text(response)


# String matching on model response
def post_process_classification(classification, ground_truth):
    if 'non-sensitive' in classification.lower():
        if ground_truth == 0:
            return 'TN', 0
        else:
            return 'FN', 0

    elif 'sensitive' in classification.lower() and 'non-sensitive' not in classification.lower():
        if ground_truth == 1:
            return 'TP', 1
        else:
            return 'FP', 1

    else:
        # Further processing required
        return classification, None
        further_processing_required[sample[1].doc_id] = classification


def clear_memory():
    # Prevents cuda out of memory
    torch.cuda.empty_cache()
    gc.collect()


# Dataset - dataframe, prompt_strategy - prompt function name, model - LLM
def llm_experiment(dataset, prompt_strategy, model, tokenizer):
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
        if len(sample_text) > 12000:
            continue

        classification = prompt_to_reply(sample_text, prompt_strategy, model, tokenizer)
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

