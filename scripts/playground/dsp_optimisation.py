import numpy as np
import sys
import os
import dspy
from huggingface_hub import login
sys.path.append("../")
from dataset import load_sara
from model import llm_experiment, post_process_split_docs
from preprocess_sara import proccutit, full_preproc
import json
import time
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from DSPyCORPO import *
import math
import pandas as pd
from transformers import AutoTokenizer
from models import get_model_version
from hf_dspy_edit import HFModelEdit
import torch
import gc

negative = '["non-sensitive"]'
positive = '["sensitive"]'

#login('###')

class SensSignature(dspy.Signature):
    """[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. You will be given an email message from a work email.
Which one of the attributes: "sensitive", "non-sensitive" describes the given message?
Always answer with a Python list containing the appropriate attribute. <<SYS>> [/INST] """

    message = dspy.InputField(desc="an email message containing potentially sensitive personal information", prefix="message:")
    answer = dspy.OutputField(desc="you answer with sensitive or not sensitive", prefix="message classification:")

class PromptNN(dspy.Module):
    def __init__(self, config):
        super().__init__()

        self.signature = SensSignature
        x = dspy.OutputField(
            desc="you reason with two short sentences so you can generate an answer",
        )
        self.predictor = dspy.ChainOfThought(SensSignature, activated=False) #, rationale_type=x)
        self.config = config

    def forward(self, document):
        result = self.predictor(message=document, **self.config)
        #print(result)
        return dspy.Prediction(
            answer=result.answer,
        )


def clear_memory():
    # Prevents cuda out of memory
    torch.cuda.empty_cache()
    gc.collect()

class vanillaNN(dspy.Module):
    def __init__(self, config):
        super().__init__()

        self.signature = SensSignature
        x = dspy.OutputField(
            desc="you reason with two short sentences so you can generate an answer",
        )
        self.predictor = dspy.ChainOfThought("question -> classification", activated=False) #, rationale_type=x)
        self.config = config

    def forward(self, question):
        result = self.predictor(question=question, **self.config)
        #print(result)
        return dspy.Prediction(
            classification=result.classification,
        )

def stream_ans(gen_ans):
    gen_ans = gen_ans.split('classification:')
    gen_ans = gen_ans[-1]
    match_string = gen_ans.lower()
    print(match_string)
    if 'non-sensitive' in match_string:
        return negative
    elif 'sensitive' in match_string:
        return positive
    else:
        return 'no answer'

def evaluation_metric(example, pred, trace=None):
    clear_memory()
    ground_truth = example.classification
    processed_pred = stream_ans(pred.classification)
    pred.classification = processed_pred
    answer_EM = ground_truth == processed_pred
    #answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    print(answer_EM)
    return answer_EM

def dspy_dataset(df):
    devset = []
    devset_with_input = []
    for s in df.iterrows():
        if s[1]["sensitivity"] == 0:
            ans = negative
        else:
            ans = positive
        instructions = """[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. You will be given an email message from a work email.
Which one of the attributes: "sensitive", "non-sensitive" describes the given message?
Always answer with a Python list containing the appropriate attribute. <<SYS>> [/INST] \nMessage:"""
        f = dspy.Example({"question": s[1]["text"], "classification": ans})
        devset.append(f)
        f = dspy.Example({"question": instructions+s[1]["text"], "classification": ans}).with_inputs("question")
        devset_with_input.append(f)
    return devset, devset_with_input


m = 'mistralai/Mistral-7B-Instruct-v0.2'
tokenizer = AutoTokenizer.from_pretrained(m, use_fast=True)
turbo = HFModelEdit(model = 'mistralai/Mistral-7B-Instruct-v0.2')
dspy.settings.configure(lm=turbo)
config = {'config': {'do_sample':False, 'max_new_tokens':10} }
generate_answer = PromptNN(config)
s = load_sara()
s = s.sample(n=20, random_state=1)
sara_df = full_preproc(s, tokenizer)
devset, devset_with_input = dspy_dataset(sara_df)
evaluator = Evaluate(devset=devset_with_input[10:], metric=evaluation_metric, num_threads=1, display_progress=True) #, display_table=0)
#evaluator(generate_answer)


#vanilla = dspy.Predict("question -> classification"config)
#evaluator(vanilla)
#vanilla = dspy.ChainOfThought("question -> classification", activated=False)
#evaluator(vanilla)

vanilla = vanillaNN(config)
evaluator(vanilla)

#CoT = dspy.ChainOfThought("question -> classification") 
#evaluator(CoT)
#fewshot = dspy.LabeledFewShot(k=8).compile(vanilla, trainset=devset_with_input[:10])
tp = BootstrapFewShotWithRandomSearch(metric=evaluation_metric)
bootstrap = tp.compile(vanilla, trainset=devset_with_input[:10], valset=devset_with_input[10:])
evaluator(bootstrap)
exit(0)
config = dict(epochs=1, bf16=True, lr=5e-5)
tp = BootstrapFewShotWithRandomSearch(metric=evaluation_metric, max_bootstrapped_demos=1,
                                      num_candidate_programs=1, num_threads=1, max_labeled_demos=1,
                                      teacher_settings=dict(lm=turbo))
lhp = tp.compile(generate_answer, trainset=devset_with_input[:50], valset=devset_with_input[50:1000])

print(1)
print(lhp)

print(generate_answer)

exit(0)





'''
scores = []
for x in devset_with_input:
    pred = generate_answer(**x.inputs())
    score = evaluation_metric(x, pred)
    scores.append(score)
'''

teleprompter = COPRO(
    metric=validate_context_and_answer,
    verbose=True,
    # Larger depth necessary but memory constraints
    depth=2,
    # How many prompts suggested for each iteration
    breadth=2
)
kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=0) # Used in Evaluate class in the optimization process
print('OPTIMISE')
compiled_prompt_opt = teleprompter.compile(cot_baseline, trainset=devset_with_input, eval_kwargs=kwargs)
print('compiled prompt opt')
print(compiled_prompt_opt)
#evaluate(compiled_prompt_opt, devset=devset_with_input)

print('End')

