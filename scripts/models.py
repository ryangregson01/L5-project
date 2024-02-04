import torch
from transformers import AutoTokenizer, GenerationConfig
from config import *
from transformers import AutoModelForCausalLM


def get_l2(version, revision):
    access_token = l2_token
    model_name = version
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=access_token)
    if revision == None:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=access_token, cache_dir=my_cache)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=access_token, cache_dir=my_cache, revision=revision)
    return tokenizer, model

def get_gpt2(version, revision):
    model_name = version
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=my_cache)
    return tokenizer, model

def get_mistral(v, r):
    model_name = v
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=my_cache)
    return tokenizer, model

def get_model_version(m, v, r=None):
    models_dict = {'get_l2' : get_l2, 'get_gpt2': get_gpt2, 'get_mistral': get_mistral}
    return models_dict.get(m)(v, r)
