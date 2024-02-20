from dataset import load_sara
from preprocess_sara import proccutit
from models import get_model_version
from prompts import *
from model import llm_experiment, post_process_split_docs
import time
import numpy as np
import os
import json


sara_df = load_sara()
samp = sara_df #.sample(n=50, random_state=1)
processed_sara_df = proccutit(samp)

#print(processed_sara_df.head())
#lens = [len(text) for text in processed_sara_df.text]
#print(sorted(lens))

#tokenizer, model = get_model_version('get_l2', 'TheBloke/Llama-2-13B-chat-GPTQ', 'gptq-8bit-64g-actorder_True')
tokenizer, model = get_model_version('get_l2', "meta-llama/Llama-2-7b-chat-hf")
#tokenizer, model = get_model_version('get_mistral', "mistralai/Mistral-7B-Instruct-v0.2")
#tokenizer, model = get_model_version('get_l2', 'TheBloke/Llama-2-70B-chat-GPTQ')
prompts = ['itspersonal', 'itspersonal_2', 'itspersonalfewshot'] #['b1', 'b2', 'b1_2', 'b2_2', 'b1sys', 'b2sys', 'b1_2sys', 'b2_2sys'] #['b1', 'b2', 'b3']
#prompts = ['bfor70b', 'bfor70b_2'] #, 'b2', 'b3', 'b1_2', 'b2_2', 'b3_2']
end_prompt = '[/INST]'
model_name = 'l27b-meta' #'mist7b-mist'#'l270B-GPTQ'

for prompt in prompts:
    prompt_str = 'results/' + model_name + '/' + prompt + '/'
    prompt = get_prompt(prompt)
    print('Starting experiment')
    start = time.time()
    predictions, further_processing_required, model_responses, truths_list, preds_list = llm_experiment(processed_sara_df, prompt, model, tokenizer, end_prompt)
    end = time.time()
    duration = end-start

    new_preds, new_truths = post_process_split_docs(model_responses, further_processing_required, preds_list, sara_df)

    truth_labs = np.array(new_truths)
    preds = np.array(new_preds)
    if not os.path.exists(prompt_str):
        os.makedirs(prompt_str)
    np.savetxt(prompt_str+'truth_labs.txt', truth_labs)
    np.savetxt(prompt_str+'preds.txt', preds)
    f = open(prompt_str+"duration.txt", "w")
    f.write(str(duration))
    f.close()

    with open(prompt_str+'resp.json', 'w') as f:
        json.dump(model_responses, f, indent=2)

print('DONE')
