import time
import numpy as np
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import sys
sys.path.append("../")
from models import get_model_version


def generate_answer(sens_aware):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(sens_aware, return_tensors='pt')
    inputs = inputs.to(d)
    generation_config = GenerationConfig(
        max_new_tokens=400,
        pad_token_id=tokenizer.pad_token_id,
    )
    output = model.generate(inputs=inputs.input_ids, attention_mask=inputs.attention_mask, generation_config=generation_config)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(decoded_output)
    print()


model_map = {'l27b-meta': ['get_l2', 'meta-llama/Llama-2-7b-chat-hf', 'main'],
            'mist7b-mist': ['get_model', 'mistralai/Mistral-7B-Instruct-v0.2', 'main'],
            'mixt-4bit': ['get_model', 'TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ', 'main'],
            }

m, v, r = model_map.get('l27b-meta')
d = 'auto'
tokenizer, model = get_model_version(m, v, r, d)
if d == 'auto':
    d = 'cuda'

sens_query = "[INST] What does sensitive personal information mean? [/INST]"
#generate_answer(sens_query)

foia_query = "[INST] What personal information is exempt under Section 40 of the Freedom of Information Act in the United Kingdom? [/INST]"
#generate_answer(foia_query)

foia_query2 = "[INST] Describe UK FOIA Section 40. [/INST]"
generate_answer(foia_query2)
