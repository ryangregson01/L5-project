import numpy as np
import sys
import os
import dspy
from huggingface_hub import login
sys.path.append("../")
from dataset import load_sara
from model import llm_experiment, post_process_split_docs
from preprocess_sara import full_preproc
import json
import time
from dspy.evaluate import Evaluate
#from dspy.teleprompt import SignatureOptimizer, COPRO
from DSPyCORPO import *
import math
import pandas as pd
from transformers import AutoTokenizer
from sig_classes import *

def write_responses_json(results, filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
    except:
        data = []
    results = data + results
    with open(filename, 'w') as file:
        json.dump(results, file, indent=4)


def main_experiment(NN, sig, break_p):
    model_name = "mistralai/Mistral-7B-Instruct-v0.2" #"TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ" ##"mistralai/Mistral-7B-Instruct-v0.2"
    turbo = dspy.HFModel(model = model_name) #"meta-llama/Llama-2-7b-chat-hf")
    dspy.settings.configure(lm=turbo)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    s = load_sara()
    p = full_preproc(s, tokenizer)
    config = {'config': {'do_sample':False, 'max_new_tokens':6} }
    generate_answer = NN(config, sig)
    ans_prefix = generate_answer.signature.fields.get('answer').json_schema_extra.get('prefix')
    #print(ans_prefix)

    mrs = []
    count_trace = 0
    for i, row in enumerate(p.iterrows()):
        row_id = row[1].doc_id
        row_text = row[1].text + '. \n [/INST] '
        row_gt = row[1].sensitivity

        gen_pred = generate_answer(document=row_text)
        ans_split = gen_pred.answer.split(ans_prefix)
        gen_ans = ans_split[-1]

        match_string = gen_ans.lower()
        if 'non-personal' in match_string:
            pred = 0
        else:
            pred = 1

        res = {
            'doc_id': row_id,
            'generated_response': gen_ans,
            'prediction': pred,
            'ground_truth': row_gt
            #'full_response': gen_pred.answer
        }
        
        mrs.append(res)
        count_trace += 1
        if (count_trace % 100) == 0:
            print(count_trace)

        if i == break_p:
            break

    return mrs

class PromptNN(dspy.Module):
    def __init__(self, config, sig):
        super().__init__()

        self.signature = sig
        x = dspy.OutputField(
            desc="you reason with two short sentences so you can generate an answer",
        )
        self.predictor = dspy.ChainOfThought(self.signature, activated=False) #, rationale_type=x)
        self.config = config

    def forward(self, document):
        result = self.predictor(message=document, **self.config)
        print(result.answer)
        return dspy.Prediction(
            answer=result.answer,
        )




mrs = main_experiment(PromptNN, base, 1)
#print(mrs)
write_responses_json(mrs, 'results/dspy_new.json')

