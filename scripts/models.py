import torch
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from config import *
from transformers import AutoModelForSeq2SeqLM

def get_flan(v, r, d):
    model_name = v
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=d, cache_dir=my_cache, revision=r, trust_remote_code=False)
    return tokenizer, model

def get_l2(version, revision, device):
    access_token = l2_token
    model_name = version
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=access_token, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, token=access_token, cache_dir=my_cache, revision=revision)
    return tokenizer, model


def get_model(v, r, d):
    model_name = v
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=d, cache_dir=my_cache, revision=r, trust_remote_code=False)
    return tokenizer, model


def get_model_version(m, v, r, d):
    models_dict = {'get_l2' : get_l2, 'get_model': get_model, 'get_flan': get_flan}
    return models_dict.get(m)(v, r, d)
