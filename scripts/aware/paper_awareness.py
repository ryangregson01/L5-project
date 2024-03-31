import time
import numpy as np
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import sys
sys.path.append("../")
from models import get_model_version
#from pipeline_config import model_map

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

sens_aware = "<s> [INST] What does sensitive personal information mean? [/INST]"
sens_aware = "<s> [INST] Would you regard names as sensitive personal information? [/INST]"
sens_aware = "[INST] Would you regard names as sensitive personal information? How should I prompt you to not consider names as sensitive personal information for a classification task. [/INST]"


method_id = 0
model_id = 0
m, v, r = model_map.get('mist7b-mist')
d = 'cpu'
tokenizer, model = get_model_version(m, v, r, d)


tokenizer.add_special_tokens({"pad_token": "<pad>"})
if tokenizer.pad_token is None:
    tokenizer.pad_token = '[PAD]' #"<pad>" #'[PAD]' #tokenizer.eos_token
inputs = tokenizer(sens_aware, return_tensors='pt')
inputs = inputs.to(d)
generation_config = GenerationConfig(
    max_new_tokens=50,
    pad_token_id=tokenizer.pad_token_id,
)
output = model.generate(inputs=inputs.input_ids, attention_mask=inputs.attention_mask, generation_config=generation_config)


decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print('Model name:', m)
print(decoded_output)
print()
