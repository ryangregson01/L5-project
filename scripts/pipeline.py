from dataset import load_sara
from preprocess_sara import get_preprocessed_sara
from models import get_model, get_model_version
from prompts import *
from model import llm_experiment
import time
import numpy as np
import os
import json


sara_df = load_sara()
# Sample before preprocessing, otherwise split documents make less sense.
processed_sara_df = get_preprocessed_sara(sara_df)
preprocessed_sara = processed_sara_df.sample(frac=0.2, random_state=1)

tokenizer, model = get_model('get_meta_l2')
prompts = ['base', 'persona', 'cot']
#prompt = get_prompt('base_prompt_template')

for prompt in prompts:
    prompt_str = 'results/' + prompt + '/'
    prompt = get_prompt(prompt)
    print('Starting experiment')
    start = time.time()
    predictions, further_processing_required, model_responses, truths_list, preds_list = llm_experiment(preprocessed_sara, prompt, model, tokenizer)
    end = time.time()
    duration = end-start

    truth_labs = np.array(truths_list)
    preds = np.array(preds_list)
    if not os.path.exists(prompt_str):
        os.makedirs(prompt_str)
    np.savetxt(prompt_str+'truth_labs.txt', truth_labs)
    np.savetxt(prompt_str+'preds.txt', preds)
    f = open(prompt_str+"duration.txt", "w")
    f.write(str(duration))
    f.close()

    with open(prompt_str+'resp.json', 'w') as f:
        json.dump(model_responses, f, indent=2)

