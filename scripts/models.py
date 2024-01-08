import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from config import *


def get_meta_l2():
    access_token = l2_token
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", use_auth_token=access_token, cache_dir=my_cache)
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
