import numpy as np
import sys
import os
import dspy
from huggingface_hub import login

login('###')
turbo = dspy.HFModel(model = "mistralai/Mistral-7B-Instruct-v0.2") #"meta-llama/Llama-2-7b-chat-hf")
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts, do_sample=False)

sys.path.append("../")
from dataset import load_sara
from model import llm_experiment, post_process_split_docs
from models import get_model_version
from preprocess_sara import proccutit
import json

import time

def write_responses_json(results, filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
    except:
        data = []
   
    results = data + results
    with open(filename, 'w') as file:
        json.dump(results, file, indent=4)

class NewBasicQA(dspy.Signature):
    """Classify text that may contain personal information."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="Does the message contain personal information?", prefix="Personal[Yes/No]:")

def prompt(d):
    return f"""Your task is to determine if the email message from a work email contains personal information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment.

Message: {d}"""

s = load_sara()
p = proccutit(s)
t = p.iloc[0].text
Q = prompt(t)
generate_answer = dspy.Predict(NewBasicQA)
c = {'config': {'do_sample':False, 'max_new_tokens':50} }
c = {'config': {'do_sample':False, 'max_new_tokens':3} }

'''
for i, t in enumerate(p.text):
    Q = prompt(t)
    pred = generate_answer(question=Q, **c)
    #print(f"Predicted Answer: {pred.answer}")#
    ans_split = pred.answer.split('Answer:')
    gen_ans = ans_split[-1]
    print(f"Response: {gen_ans}")
    
    if i == 3:
        break
'''

start = time.time()
mrs = []
count_trace = 0
for row in p.iterrows():
    row_id = row[1].doc_id
    row_text = row[1].text
    row_gt = row[1].sensitivity

    Q = prompt(t)
    pred = generate_answer(question=Q, **c)
    ans_split = pred.answer.split('Personal[Yes/No]:')
    gen_ans = ans_split[-1]

    match_string = gen_ans.lower()
    if 'no' in match_string:
        pred = 0
    elif 'yes' in match_string:
        pred = 1
    else:
        pred = 2

    res = {
        'doc_id': row_id,
        'generated_response': gen_ans,
        'prediction': pred,
        'ground_truth': row_gt
    }
    
    mrs.append(res)
    count_trace += 1
    if (count_trace % 100) == 0:
        print(count_trace)

end = time.time()
duration = end-start
print("Duration", duration)
write_responses_json(mrs, 'results/dsp10mist.json')

