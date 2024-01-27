import torch
from transformers import AutoTokenizer, GenerationConfig
from config import *
from transformers import AutoModelForCausalLM


def get_meta_l2():
    access_token = l2_token
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=access_token, cache_dir=my_cache)
    return tokenizer, model

def get_thebloke_l2():
    access_token = l2_token
    model_name = "TheBloke/Llama-2-7B-Chat-GPTQ"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", use_auth_token=access_token, cache_dir=my_cache)
    return tokenizer, model

def get_model(model_name):
    models_dict = {'get_meta_l2' : get_meta_l2, 'get_thebloke_l2': get_thebloke_l2}
    return models_dict.get(model_name)()

def get_l2(version):
    access_token = l2_token
    model_name = version
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=access_token, cache_dir=my_cache)
    return tokenizer, model

def get_gpt2(version):
    model_name = version
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=my_cache)
    return tokenizer, model

def get_mistral(v):
    model_name = v
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=my_cache)
    return tokenizer, model

def get_model_version(m, v):
    models_dict = {'get_l2' : get_l2, 'get_gpt2': get_gpt2, 'get_mistral': get_mistral}
    return models_dict.get(m)(v)
