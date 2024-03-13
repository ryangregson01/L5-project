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
from dspy.datasets import HotPotQA
#from dspy.teleprompt import SignatureOptimizer, COPRO
from DSPyCORPO import *


login('###')

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
            desc="Sentence classification: ${class}",
        )
        self.predictor = dspy.ChainOfThought(self.signature, activated=False) #, rationale_type=x)
        self.config = config

    def forward(self, question):
        result = self.predictor(question=question, **self.config)
        return dspy.Prediction(
            answer=result.answer,
        )


def stream_ans(gen_ans):
    gen_ans = gen_ans.split('classification:')
    gen_ans = gen_ans[-1]
    match_string = gen_ans.lower()
    if 'not sensitive' in match_string:
        return 'not sensitive'
    elif 'sensitive' in match_string:
        return 'sensitive'
    else:
        return 'no answer'

def validate_context_and_answer(example, pred, trace=None):
    ground_truth = example.answer
    processed_pred = stream_ans(pred.answer)
    pred.answer = processed_pred
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    return answer_EM

turbo = dspy.HFModel(model = 'mistralai/Mistral-7B-Instruct-v0.2')
dspy.settings.configure(lm=turbo)
sara_df = proccutit(load_sara())
#sara_df = sara_df.sample(n=10, random_state=1)

devset = []
devset_with_input = []

for s in sara_df.iterrows():
    if s[1]["sensitivity"] == 0:
        ans = 'not sensitive'
    else:
        ans = 'sensitive'
    f = dspy.Example({"question": s[1]["text"], "answer": ans})
    devset.append(f)
    f = dspy.Example({"question": s[1]["text"], "answer": ans}).with_inputs("question")
    devset_with_input.append(f)

NUM_THREADS = 5
evaluate = Evaluate(devset=devset, metric=validate_context_and_answer, num_threads=NUM_THREADS, display_progress=True, display_table=False)
config = {'config': {'do_sample':False, 'max_new_tokens':50}}
cot_baseline = PromptNN(config)
#print(evaluate.devset[i].question)
#print(cot_baseline(evaluate.devset[0].question))
evaluate(cot_baseline, devset=devset_with_input)

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

