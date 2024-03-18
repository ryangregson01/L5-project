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

negative = 'not sensitive'
positive = 'sensitive'

#login('###')

class SensSignature(dspy.Signature):
    """You are given some context (what sensitive personal information is), a message (what you must classify) and a question (what you must answer). You must answer with sensitive/not sensitive in response to the question given the context and message."""
    
    context = dspy.InputField(desc="", prefix="The message is from a work email and may contain sensitive personal information. Messages with sensitive personal information are purely personal and are unrelated to work. All other professional work-related messages are not sensitive.")
    message = dspy.InputField(prefix="Message:")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="sensitive or not sensitive", prefix="Message classification:")

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
        result = self.predictor(message=question, question="Does the message contain personal sensitive information? Classify the message among sensitive, not sensitive.", **self.config)
        return dspy.Prediction(
            answer=result.answer,
        )


def stream_ans(gen_ans):
    gen_ans = gen_ans.split('classification:')
    gen_ans = gen_ans[-1]
    match_string = gen_ans.lower()
    if 'not sensitive' in match_string:
        return negative
    elif 'sensitive' in match_string:
        return positive
    else:
        return 'no answer'

def evaluation_metric(example, pred, trace=None):
    ground_truth = example.answer
    processed_pred = stream_ans(pred.answer)
    print(processed_pred)
    pred.answer = processed_pred
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    return answer_EM

def dspy_dataset(df):
    devset = []
    devset_with_input = []
    for s in df.iterrows():
        if s[1]["sensitivity"] == 0:
            ans = negative
        else:
            ans = positive
        f = dspy.Example({"question": s[1]["text"], "answer": ans})
        devset.append(f)
        f = dspy.Example({"question": s[1]["text"], "answer": ans}).with_inputs("question")
        devset_with_input.append(f)
    return devset, devset_with_input


turbo = dspy.HFModel(model = 'mistralai/Mistral-7B-Instruct-v0.2')
dspy.settings.configure(lm=turbo)
config = {'config': {'do_sample':False, 'max_new_tokens':10} }
generate_answer = PromptNN(config)
sara_df = full_preproc(load_sara())
#sara_df = sara_df.sample(n=50, random_state=1)
devset, devset_with_input = dspy_dataset(sara_df)
evaluator = Evaluate(devset=devset_with_input, metric=evaluation_metric, num_threads=1, display_progress=True) #, display_table=0)
#evaluator(generate_answer)

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

