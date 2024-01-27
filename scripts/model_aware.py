from models import get_model, get_model_version
from model import llm_experiment
import time
import numpy as np
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


sens_aware = "<s> [INST] What does sensitive personal information mean? [/INST]"
method_id = 0
model_id = 0
methods = ['get_l2']
models = ["meta-llama/Llama-2-7b-chat-hf", "TheBloke/Llama-2-7B-Chat-GPTQ", "TheBloke/Llama-2-13B-Chat-GPTQ", "TheBloke/Llama-2-70B-Chat-GPTQ"]

for i in range(4):
    tok, mod = get_model_version(methods[i], models[i])
    inputs = tok(sens_aware, return_tensors='pt')
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=300
    )
    output = mod.generate(inputs=inputs.input_ids.cuda(), attention_mask=inputs.attention_mask.cuda())#, generation_config=generation_config)
    decoded_output = tok.decode(output[0], skip_special_tokens=True)

    print('Model name:', models[model_id])
    print(decoded_output)
    print()
