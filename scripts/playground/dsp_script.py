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

login('###')

def write_responses_json(results, filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
    except:
        data = []
    results = data + results
    with open(filename, 'w') as file:
        json.dump(results, file, indent=4)


def main_experiment(NN, end_prompt, break_p):
    turbo = dspy.HFModel(model = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ") #"mistralai/Mistral-7B-Instruct-v0.2") #"meta-llama/Llama-2-7b-chat-hf")
    dspy.settings.configure(lm=turbo)

    s = load_sara()
    p = proccutit(s)
    config = {'config': {'do_sample':False, 'max_new_tokens':10} }
    generate_answer = NN(config)

    mrs = []
    count_trace = 0
    for i, row in enumerate(p.iterrows()):
        row_id = row[1].doc_id
        row_text = row[1].text
        row_gt = row[1].sensitivity

        if len(row_text) > 9500:
            continue

        gen_pred = generate_answer(question=row_text)
        ans_split = gen_pred.answer.split(end_prompt)
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

        #if i == break_p:
        #    break

    return mrs


class SensSignature(dspy.Signature):
    # Prompt instructions
    """The text is from a work email and may contain sensitive personal information. Classify text among sensitive, not sensitive."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="The classification label, either 'sensitive' or 'not sensitive'.")

class PromptNN(dspy.Module):
    def __init__(self, config):
        super().__init__()

        self.signature = SensSignature
        self.predictor = dspy.Predict(self.signature)
        self.config = config

    def forward(self, question):
        result = self.predictor(question=question, **self.config)
        return dspy.Prediction(
            answer=result.answer,
        )

mrs = main_experiment(PromptNN, 'Answer:', 1)
#print(mrs)
write_responses_json(mrs, 'results/mixt4bit.json')

