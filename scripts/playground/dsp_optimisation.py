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
from dspy.teleprompt import SignatureOptimizer


login('###')

class SensSignature(dspy.Signature):
    """The text is from a work email and may contain sensitive personal information. Classify text among sensitive, not sensitive."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="The classification label, either 'sensitive' or 'not sensitive'.")

class CoTSignature(dspy.Signature):
    """Answer the question."""

    question = dspy.InputField(desc="question about something")
    answer = dspy.OutputField(desc="often between 1 and 5 words")

class PromptNN(dspy.Module):
    def __init__(self, config):
        super().__init__()

        self.signature = CoTSignature
        # https://github.com/stanfordnlp/dspy/issues/386 - cannot use Predict class w optimiser, must use CoT
        # Must pass in OutputField as rational_type with empty content to surpass CoT default styling
        x = dspy.OutputField(
            prefix="",
            desc="",
        )
        self.predictor = dspy.ChainOfThought(self.signature, rationale_type=x)
        self.config = config

    def forward(self, question):
        result = self.predictor(question=question, **self.config)
        return dspy.Prediction(
            answer=result.answer,
        )


def validate_context_and_answer(example, pred, trace=None):
    s = pred.answer.split('Answer: ')
    pred.answer = s[-1]
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    return answer_EM


#hf_device_map = 'cuda:0'
turbo = dspy.HFModel(model = 'mistralai/Mistral-7B-Instruct-v0.2') #, hf_device_map=hf_device_map)
dspy.settings.configure(lm=turbo)
dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=20, test_size=0)
trainset, devset = dataset.train, dataset.dev
NUM_THREADS = 5
evaluate = Evaluate(devset=devset, metric=validate_context_and_answer, num_threads=NUM_THREADS, display_progress=True, display_table=False)
config = {'config': {'do_sample':False, 'max_new_tokens':250} }
cot_baseline = PromptNN(config)
devset_with_input = [dspy.Example({"question": r["question"], "answer": r["answer"]}).with_inputs("question") for r in devset]
#evaluate(cot_baseline, devset=devset_with_input)

teleprompter = SignatureOptimizer(
    metric=validate_context_and_answer,
    verbose=False,
    # Larger depth necessary but memory constraints
    depth=1,
    # How many prompts suggested for each iteration
    breadth=2
)
kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=0) # Used in Evaluate class in the optimization process
print('OPTIMISE')
compiled_prompt_opt = teleprompter.compile(cot_baseline, devset=devset_with_input, eval_kwargs=kwargs)
print('compiled prompt opt')
print(compiled_prompt_opt)
#evaluate(compiled_prompt_opt, devset=devset_with_input)

print('End')

