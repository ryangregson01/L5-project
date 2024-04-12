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





def task(): return """Your task is to determine if the following email message contains sensitive personal information that should be protected, or is a non-personal email."""

def collection_context(): return """You are reviewing emails from the company Enron that are related to the California energy crisis, trading discrepancies, and other messages occurring near the end of the company's life."""

def question_config(): return """Which one of the attributes: "personal", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute."""

def message(d): return f"""Message: {d}. """

def answer(): return """Answer: ["""

def sens_cat_info(): return """Personal messages messages may be purely personal and do not include any relation to work being done. Personal messages that are related to work will contain comments about the quality of people's work and expressions of feelings about employee treatment."""

def non_sens_cat_info(): return """Non-personal messages discuss topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support), employment arrangements (job seeking, hiring, recommendations), document editing/checking (collaboration), empty message (due to missing attachment), empty message."""

def all_cats_system():
    return f"""[INST] <<SYS>> 
{task()}

{collection_context()}

{sens_cat_info()} {non_sens_cat_info()}

{question_config()} <</SYS>> """

class AllCatsSignaturealt(dspy.Signature):
    """Identify sensitive personal information within email messages"""

    system = dspy.InputField()
    message = dspy.InputField(desc="a email message", prefix="Message:")
    answer = dspy.OutputField(desc='you answer with ["personal"] or ["non-personal"]', prefix="Answer:")

class AllCatsSignature(dspy.Signature):
    """[INST] <<SYS>> 
Your task is to determine if the following email message contains sensitive personal information that should be protected, or is a non-personal email.

You are reviewing emails from the company Enron that are related to the California energy crisis, trading discrepancies, and other messages occurring near the end of the company's life.

Personal messages messages may be purely personal and do not include any relation to work being done. Personal messages that are related to work will contain comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages discuss topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support), employment arrangements (job seeking, hiring, recommendations), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "personal", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> """

    message = dspy.InputField(desc="a email message", prefix="Message:")
    answer = dspy.OutputField(desc='you answer with ["personal"] or ["non-personal"]', prefix="Answer:")

class sens_cat_sens_few(dspy.Signature):
    """[INST] <<SYS>> 
Your task is to determine if the following email message contains sensitive personal information that should be protected, or is a non-personal email.

You are reviewing emails from the company Enron that are related to the California energy crisis, trading discrepancies, and other messages occurring near the end of the company's life.

You are protecting sensitive personal information that would be exempt under Section 40 of the Freedom of Information Act in the United Kingdom.

Personal messages messages may be purely personal and do not include any relation to work being done. Personal messages that are related to work will contain comments about the quality of people's work and expressions of feelings about employee treatment.

Which one of the attributes: "personal", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Example:
Message: i am so sorry you guys gave him a really good life robert kean on pm please respond to to pat hilleman e mail greg smith e mail greg smith e mail colin lamb e mail maggie kean e mail nora kean e mail nora kean e mail sann evelyn gossum e mail doug karen reiman e mail jean kean e mail kathy wedig e mail melissa kean e mail phil kean e mail steve melissa kean e mail cc subject houston houston died today after a long battle with auto immune hemolytic anemia rest in peace rob kean.
Answer: ["personal"].

Example:
Message: original message from alvarez ray sent thursday august pm to sanders richard b williams robert c cc steffes james d comnes alan robertson linda shapiro richard subject california refund proceeding privileged and confidential attorney work product attorney client communication in our litigation meeting last tuesday we learned that netting of spot purchases and sales is of paramount importance in limiting our exposure to refunds under the fercs july order in both the california and pacific northwest proceedings however in the california proceeding we are also afforded the opportunity in the order to offset our receivables against any refund that would otherwise have to be paid by us also to the extent that our receivables exceed the refund amount the order provides for payment of interest to us in accordance with cfr section a at present i am aware of the existence of three receivables as you will recall i mentioned at tuesdays meeting that we should be able to treat as a receivable for purposes of offset the revenues due us from the application of the caiso underscheduling penalty i communicated the possibility of offsetting this receivable against any potential refund at this mornings western wholesale conference call and subsequent research conducted by alan comnes indicates that the caiso owes us mm as per the caisos own numbers the caiso also owes us approximately mm due to nonpayment for one to two months of business these two figures alone totalling approximately mm exceed the estimate of our potential exposure in the california proceeding finally the negative ctc issue is another receivable due us by ious that under the plain language of the order could potentially be used to offset any refund amount i am not aware of the magnitude of this receivable but do understand that there are some sensitivities regarding this issue including concerns about revealing the amount of the credit due us it is likely that the amount would become public if it is presented in the california proceeding please give us your thoughts as to the use of the ctc receivable to offset potential refunds in the california proceeding.
Answer: ["non-personal"].

Now answer:"""

    message = dspy.InputField(prefix="Message:")
    answer = dspy.OutputField(prefix="Answer:")
    #answer = dspy.OutputField(desc='you answer with ["personal"] or ["non-personal"]', prefix="Answer:")


class base(dspy.Signature):
    """[INST] <<SYS>> 
Your task is to determine if the following email message contains sensitive personal information that should be protected, or is a non-personal email.

You are reviewing emails from the company Enron that are related to the California energy crisis, trading discrepancies, and other messages occurring near the end of the company's life.

Which one of the attributes: "personal", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> """

    message = dspy.InputField(prefix="Message:")
    answer = dspy.OutputField(prefix="Answer:")
    #answer = dspy.OutputField(desc='you answer with ["personal"] or ["non-personal"]', prefix="Answer:")



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




mrs = main_experiment(PromptNN, base, 2000)
#print(mrs)
#for l in mrs:
#    print(l.get('generated_response')[:10])
write_responses_json(mrs, 'results/base_dspy.json')

