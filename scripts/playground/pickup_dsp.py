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
    model_name = "mistralai/Mistral-7B-Instruct-v0.2" ##"mistralai/Mistral-7B-Instruct-v0.2"
    turbo = dspy.HFModel(model = model_name) #"meta-llama/Llama-2-7b-chat-hf")
    dspy.settings.configure(lm=turbo)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    s = load_sara()
    p = full_preproc(s, tokenizer)
    config = {'config': {'do_sample':False, 'max_new_tokens':60} }
    generate_answer = NN(config, sig)
    ans_prefix = generate_answer.signature.fields.get('answer').json_schema_extra.get('prefix')
    #print(ans_prefix)

    mrs = []
    count_trace = 0
    for i, row in enumerate(p.iterrows()):
        row_id = row[1].doc_id
        row_text = row[1].text + '.'
        row_gt = row[1].sensitivity

        gen_pred = generate_answer(document=row_text)
        ans_split = gen_pred.answer.split(ans_prefix)
        gen_ans = ans_split[-1]

        match_string = gen_ans.lower()
        if 'non-sensitive' in match_string:
            pred = 0
        #elif 'sensitive' in match_string:
        #    pred = 1
        else:
            pred = 1

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
    """you are given some context about what sensitive personal information is, the question you must answer, and a message. You must answer with sensitive/not sensitive to answer the question given the context and message."""
    
    context = dspy.InputField()
    question = dspy.InputField()
    message = dspy.InputField(desc="a potentially sensitive email message", prefix="message:")
    answer = dspy.OutputField(desc="you answer with sensitive or not sensitive", prefix="message classification:")

class PromptNN(dspy.Module):
    def __init__(self, config, sig):
        super().__init__()

        self.signature = SensSignature
        x = dspy.OutputField(
            desc="you reason with two short sentences so you can generate an answer",
        )
        self.predictor = dspy.ChainOfThought(self.signature, activated=False) #, rationale_type=x)
        self.config = config
        self.context = "Your task is to determine if the message from a work email is purely personal, personal but in a professional context, or non-personal. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information. If the message is non-personal, you should classify the message as not sensitive, otherwise purely personal and personal but in a professional context messages should be classified as sensitive."
        self.question="Does the message contain sensitive personal information? Classify the message among sensitive, not sensitive."

    def forward(self, document):
        result = self.predictor(context=self.context.lower(), question=self.question.lower(), message=document, **self.config)
        #print(result)
        return dspy.Prediction(
            answer=result.answer,
        )


class SensSignature(dspy.Signature):
    """your task is to determine if the message from a work email is purely personal, personal but in a professional context, or non-personal. purely personal messages include personal information and do not include any relation to work being done. personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. non-personal messages are professional emails that do not include personal information. if the message is non-personal, you should classify the message as not sensitive, otherwise purely personal and personal but in a professional context messages should be classified as sensitive. does the message contain sensitive personal information? classify the message among sensitive, not sensitive."""

    message = dspy.InputField(desc="a potentially sensitive email message", prefix="message:")
    answer = dspy.OutputField(desc="you answer with sensitive or not sensitive", prefix="message classification:")

class SensSignature2(dspy.Signature):
    """your task is to determine if the message from a work email is purely personal, personal but in a professional context, or non-personal. purely personal messages include personal information and do not include any relation to work being done. personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. non-personal messages are professional emails that do not include personal information. if the message is non-personal, you should classify the message as not sensitive, otherwise purely personal and personal but in a professional context messages should be classified as sensitive. does the message contain sensitive personal information? classify the message among sensitive, not sensitive."""

    message = dspy.InputField(desc="a potentially sensitive email message", prefix="message:")
    answer = dspy.OutputField(desc="you answer with sensitive or not sensitive", prefix="message classification:")


class PromptNN(dspy.Module):
    def __init__(self, config, sig):
        super().__init__()

        self.signature = SensSignature
        x = dspy.OutputField(
            desc="you reason with two short sentences so you can generate an answer",
        )
        self.predictor = dspy.ChainOfThought(SensSignature2, activated=True, rationale_type=x)
        #self.predictor2 = dspy.ChainOfThought(self.signature, activated=False) #, rationale_type=x)
        self.config = config

    def forward(self, document):
        result = self.predictor(message=document, **self.config)
        #print(result)
        res = ''
        #result2 = self.predictor2(message=res)
        return dspy.Prediction(
            answer=result.answer,
        )



class pdc(dspy.Signature):
    """[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails discussing topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "purely personal", "personal but in a professional context", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> [/INST] """

    message = dspy.InputField(desc="an email message", prefix="Message:")
    answer = dspy.OutputField(desc='answer with "purely personal", "personal but in a professional context", or "non-personal"', prefix="Answer:")

class hop1(dspy.Signature):
    """[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains personal information.

State what personal information is present in the message if any. <</SYS>> [/INST] """

    message = dspy.InputField(desc="an email message", prefix="Message:")
    answer = dspy.OutputField(desc='describe present personal information', prefix="Answer:")

class hop2(dspy.Signature):
    """[INST] <<SYS>> 
Your task is to determine if the email message from a work email belongs to a personal or non-personal email topic.

You have already identified if personal information could be present within the message, which is appended to the message.

Personal messages discuss purely personal or personal in a professional context information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails discussing topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support), employment arrangements (job seeking, hiring, recommendations), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

State if the personal information you identified is purely personal, personal but in a professional context, or belongs to a non-personal email topic. <</SYS>> [/INST] """

    message = dspy.InputField(desc="an email message", prefix="Message:")
    identified = dspy.InputField(desc="personal information identified")
    answer = dspy.OutputField(desc='reason about email topic', prefix="Answer:")

class hop3(dspy.Signature):
    """[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal information.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails discussing topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

You have already identified if personal information is present within the message, and then what email topic this message belongs to. This reasoning is shown under the message.

Which one of the attributes: "purely personal", "personal but in a professional context", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute <</SYS>> [/INST] """

    message = dspy.InputField(desc="an email message", prefix="Message:")
    reasoning = dspy.InputField(desc="reasoning for answer", prefix="Reasoning:")
    answer = dspy.OutputField(desc='answer with "purely personal", "personal but in a professional context", or "non-personal"', prefix="Answer:")


class pdcNN(dspy.Module):
    def __init__(self, config, sig):
        super().__init__()

        self.signature = pdc
        x = dspy.OutputField(
            desc="you reason with two short sentences so you can generate an answer",
        )
        self.hop1 = dspy.ChainOfThought(hop1, activated=False)
        self.hop2 = dspy.ChainOfThought(hop2, activated=False)
        self.hop3 = dspy.ChainOfThought(hop3, activated=False)
        self.predictor = dspy.ChainOfThought(self.signature, activated=False) #, rationale_type=x)
        self.config = config

    def forward(self, document):
        hop = self.hop1(message=document)
        #print(hop)
        #print(hop.answer)
        ans_split = hop.answer.split('Answer: ')
        gen_ans = ans_split[-1]
        #print(gen_ans)
        hop = self.hop2(message=document, identified=gen_ans)
        ans_split = hop.answer.split('Answer: ')
        gen_ans2 = ans_split[-1]
        reason = gen_ans+gen_ans2
        #print(reason)
        short_config = {'config': {'do_sample':False, 'max_new_tokens':10} }
        hop = self.hop3(message=document, reasoning=reason, **short_config)
        #print(hop)

        #result = self.predictor(message=document, **self.config)
        #print(result)
        result = hop
        return dspy.Prediction(
            answer=result.answer,
        )



mrs = main_experiment(pdcNN, pdc, 2000)
#print(mrs)
#for l in mrs:
#    print(l.get('generated_response')[:10])
write_responses_json(mrs, 'results/dsp3hopcot.json')

