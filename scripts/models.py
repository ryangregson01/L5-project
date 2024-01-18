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

def get_openai_gpt2():
    #tokenizer = AutoTokenizer.from_pretrained("gpt2")
    #model = AutoModelForCausalLM.from_pretrained("gpt2")
    #access_token = l2_token
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=my_cache)
    return tokenizer, model

def get_openai_distilgpt2():
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=my_cache)
    return tokenizer, model

def get_openai_gpt2(version):
    model_name = version
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=my_cache)
    return tokenizer, model

def get_thebloke_l270bcpu():
    access_token = l2_token
    model_name = "meta-llama/Llama-2-70b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", use_auth_token=access_token, cache_dir='/scratch2/2469038g/fake_cache')
    #model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q2_K.gguf", model_type="llama", gpu_layers=0)
    return tokenizer, model

def get_mistral_mistral7b():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=my_cache)
    return tokenizer, model


def get_model(model_name):
    models_dict = {'get_meta_l2' : get_meta_l2, 'get_thebloke_l2': get_thebloke_l2, 'get_openai_gpt2': get_openai_gpt2, 'get_openai_distilgpt2': get_openai_distilgpt2, 'get_thebloke_l270bcpu': get_thebloke_l270bcpu, 'get_mistral_mistral7b': get_mistral_mistral7b}
    return models_dict.get(model_name)()

def get_model_version(m, v):
    models_dict = {'get_meta_l2' : get_meta_l2, 'get_thebloke_l2': get_thebloke_l2, 'get_openai_gpt2': get_openai_gpt2}
    return models_dict.get(m)(v)
