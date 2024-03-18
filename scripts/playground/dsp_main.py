import numpy as np
import sys
import os
import dspy
from huggingface_hub import login
sys.path.append("../")
from dataset import load_sara
from model import llm_experiment, post_process_split_docs
from preprocess_sara import proccutit
import json
import time
from dspy.evaluate import Evaluate
#from dspy.teleprompt import SignatureOptimizer, COPRO
from DSPyCORPO import *
from jk_proc_dsp import jkproc
import math
import pandas as pd

#login('###')

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
    turbo = dspy.HFModel(model = "mistralai/Mistral-7B-Instruct-v0.2") #"meta-llama/Llama-2-7b-chat-hf")
    dspy.settings.configure(lm=turbo)

    s = load_sara()
    p = proccutit(s)
    config = {'config': {'do_sample':False, 'max_new_tokens':10} }
    generate_answer = NN(config, sig)
    ans_prefix = generate_answer.signature.fields.get('answer').json_schema_extra.get('prefix')

    mrs = []
    count_trace = 0
    for i, row in enumerate(p.iterrows()):
        row_id = row[1].doc_id
        row_text = row[1].text
        row_gt = row[1].sensitivity

        if len(row_text) > 9500:
            continue

        gen_pred = generate_answer(question=row_text)
        ans_split = gen_pred.answer.split(ans_prefix)
        gen_ans = ans_split[-1]

        match_string = gen_ans.lower()
        if 'not sensitive' in match_string:
            pred = 0
        elif 'sensitive' in match_string:
            pred = 1
        else:
            pred = 2

        res = {
            'doc_id': row_id,
            'generated_response': gen_ans,
            'prediction': pred,
            'ground_truth': row_gt,
            'full_response': gen_pred.answer
        }
        
        mrs.append(res)
        count_trace += 1
        if (count_trace % 100) == 0:
            print(count_trace)

        if i == break_p:
            break

    return mrs

#1. """The text is from a work email and may contain sensitive personal information. Classify text among sensitive, not sensitive."""
#2. """Classify an email message from a work inbox as containing sensitive personal information or not. Messages with sensitive personal information can be purely personal (unrelated to work). Additionally, they may contain sensitive personal information in a professional context, such as comments on work quality or feelings about employee treatment, while excluding discussions of company business or strategy. Assign the message to one of two categories: sensitive or not sensitive."""
class SensSignature(dspy.Signature):
    """The text is from a work email and may contain sensitive personal information. Classify text among sensitive, not sensitive."""
    
    question = dspy.InputField(prefix="Message:")
    answer = dspy.OutputField(desc="", prefix="Message classification:")

class PromptNN(dspy.Module):
    def __init__(self, config):
        super().__init__()

        self.signature = SensSignature
        x = dspy.OutputField(
            prefix="",
            desc="",
        )
        self.predictor = dspy.ChainOfThought(self.signature, activated=False) #, rationale_type=x)
        self.config = config

    def forward(self, question):
        result = self.predictor(question=question, **self.config)
        return dspy.Prediction(
            answer=result.answer,
        )


mrs = main_experiment(PromptNN, 1)
print(mrs)
#write_responses_json(mrs, 'results/mistchunks.json')

