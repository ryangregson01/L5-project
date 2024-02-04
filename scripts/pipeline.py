from dataset import load_sara
from preprocess_sara import proc3
from models import get_model, get_model_version
from prompts import *
from model import llm_experiment, post_process_split_docs
import time
import numpy as np
import os
import json


sara_df = load_sara()
samp = sara_df.sample(n=3, random_state=1)
processed_sara_df = proc3(samp)

tokenizer, model = get_model_version('get_l2', 'TheBloke/Llama-2-13B-chat-GPTQ', 'gptq-8bit-64g-actorder_True')
prompts = ['b1', 'b2', 'b3']
end_prompt = '[/INST]'

for prompt in prompts:
    prompt_str = 'results/' + prompt + '/'
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

