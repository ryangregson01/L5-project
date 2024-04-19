import time
import numpy as np
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import sys
sys.path.append("../")
from models import get_model_version
#from pipeline_config import model_map
import torch
from dataset import load_sara
from preprocess_sara import full_preproc
from final_prompts import *


def find_logits(tokenizer, outputs):
    tokens_of_interest = ['non', 'personal']
    token_probabilities = []
    generated_tokens = []
    token_ids_of_interest = {4581:'non', 28545:'personal'}
    probabilities_of_interest = {token: [] for token in tokens_of_interest}

    for logits in outputs.scores:
        probs = torch.softmax(logits, dim=-1)
        for token_id, token in token_ids_of_interest.items():
            token_probability = probs[0, token_id].item()
            probabilities_of_interest[token].append(token_probability)

        next_token_id = torch.argmax(probs, dim=-1).item()
        generated_tokens.append(next_token_id)
        token_probability = probs[0, next_token_id].item()
        token_probabilities.append(token_probability)

    k1 = probabilities_of_interest.get('non')
    k2 = probabilities_of_interest.get('personal')
    print(k1)
    found_non = 0
    found_pers = 0
    if sum(k1) > 0.5:
        for i, v in enumerate(k1):
            if v > 0.5:
                found_non = v
                found_pers = k2[i]
                break

    else:
        for i, v in enumerate(k2):
            if v > 0.5:
                found_non = k1[i]
                found_pers = v
        

    print(found_non)
    print(found_pers)

    decoded_tokens = tokenizer.decode(generated_tokens)
    print(decoded_tokens)
    return decoded_tokens, found_non

def main():
    model_map = {'l27b-meta': ['get_l2', 'meta-llama/Llama-2-7b-chat-hf', 'main'], # 2048 max tokens doc
                'l213b-8bit': ['get_l2', 'TheBloke/Llama-2-13B-chat-GPTQ', 'gptq-8bit-64g-actorder_True'], 
                'l270b-4bit': ['get_l2', 'TheBloke/Llama-2-70B-chat-GPTQ', 'main'],
                'mist7b-mist': ['get_model', 'mistralai/Mistral-7B-Instruct-v0.2', 'main'],
                'mixt-4bit': ['get_model', 'TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ', 'main'],
                'flan-large': ['get_flan', 'google/flan-t5-large', 'main'], # 320 max tokens doc
                'l2-nochat': ['get_l2', 'meta-llama/Llama-2-7b-hf', 'main'],
                'mist-awq': ['get_model', 'TheBloke/Mistral-7B-Instruct-v0.2-AWQ', 'main'],
                'l2bnb': ['get_l2_bits', 'meta-llama/Llama-2-13b-hf', 'main'],
                }


    m, v, r = model_map.get('mist-awq')
    d = 'auto'
    tokenizer, model = get_model_version(m, v, r, d)


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    p = full_preproc(load_sara(), tokenizer)

    for i, v in p.iterrows():
        input_text = all_cats(p.iloc[i].text)
        input_text = all_cats_sens(p[p.doc_id=='173153'].iloc[0].text)

        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        input_ids = input_ids.to('cuda')
        outputs = model.generate(input_ids, max_new_tokens=6, output_scores=True, return_dict_in_generate=True)
        find_logits(tokenizer, outputs)

        if i == 1:
            break


main()