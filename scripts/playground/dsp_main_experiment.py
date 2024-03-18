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
    #turbo = dspy.HFModel(model = "mistralai/Mistral-7B-Instruct-v0.2") #"meta-llama/Llama-2-7b-chat-hf")
    #dspy.settings.configure(lm=turbo)

    s = load_sara()
    p = proccutit(s)
    
    '''
    x = jkproc(s)
    #print(x.head())

    #for i, v in enumerate(x.text):
    #    print('DOCUMENT', i)
    #    print(v)
    #    print()

    proc_df = []
    doc_max_length = 2048
    for i, s in enumerate(x.iterrows()):
        idd, t, s = s[1].doc_id, s[1].text, s[1].sensitivity
        if len(t) < doc_max_length:
            proc_df.append({'doc_id':idd, 'text':t, 'sensitivity':s})
            continue
        
        chunks = math.ceil( (len(t) / doc_max_length) )
        for chunk in range(chunks):
            chunk_idd = idd + '_' + str(chunk)
            chunk_t = t[(chunk*doc_max_length):((chunk+1)*doc_max_length)]
            proc_df.append({'doc_id':chunk_idd, 'text':chunk_t, 'sensitivity':s})

    p = pd.DataFrame(proc_df)
    '''
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
##"""You must determine if a email message from a work email inbox contains sensitive personal information. Messages that include sensitive personal information can be purely personal do not include any relation to work being done. Messages that include sensitive personal information can also be related to work being done, but do not discuss company business or strategy, instead discussing sensitive personal expressions in a professional context such as comments about the quality of people's work and expressions of feelings about employee treatment. Classify the message among sensitive, not sensitive."""
class SensSignature(dspy.Signature):
    """Classify an email message from a work inbox as containing sensitive personal information or not. Messages with sensitive personal information can be purely personal (unrelated to work). Additionally, they may contain sensitive personal information in a professional context, such as comments on work quality or feelings about employee treatment, while excluding discussions of company business or strategy. Assign the message to one of two categories: sensitive or not sensitive."""

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

#Exp extravaganza
w = """Classify an email message from a work inbox as containing sensitive personal information or not. Messages with sensitive personal information can be purely personal (unrelated to work). Additionally, they may contain sensitive personal information in a professional context, such as comments on work quality or feelings about employee treatment, while excluding discussions of company business or strategy. Assign the message to one of two categories: sensitive or not sensitive."""
x = """Classify an email message from a work inbox as sensitive or not sensitive based on the presence of sensitive personal information. An email is sensitive if it includes personal information not directly related to work tasks, such as personal life details or issues. It's also sensitive if it contains personal opinions or feelings about work, like comments on work quality or sentiments about employee treatment. Note that routine mentions of names, addresses, or phone numbers in a work context, like scheduling meetings or assigning tasks, are not considered sensitive. Emails that discuss company business, strategies, or professional tasks without divulging personal or confidential information should be classified as not sensitive. Review the content carefully to determine the correct category based on these guidelines."""
y = """Classify an email message from a work inbox as sensitive or not sensitive based on the presence of sensitive personal information. An email is sensitive if it includes personal information not directly related to work tasks, such as personal life details or issues. It's also sensitive if it contains personal opinions or feelings about work, like comments on work quality or sentiments about employee treatment. Note that routine mentions of names, addresses, or phone numbers in a work context, like scheduling meetings or assigning tasks, are not considered sensitive. Emails that discuss company business, strategies, or professional tasks without divulging personal information should be classified as not sensitive. Review the content carefully to determine the correct category based on these guidelines."""
z = """Classify an email message from a work inbox as sensitive or not sensitive based on the presence of sensitive personal information. An email should be categorized as sensitive if it includes personal information unrelated to work tasks, such as personal life details or issues. Additionally, classify an email as sensitive if it contains personal opinions or feelings related to work, like comments on work quality or sentiments about employee treatment. Note that emails discussing company business, strategies, or professional tasks, without personal information or opinions, should not be considered sensitive. If an email does not meet the criteria for being sensitive, categorise it as not sensitive. Carefully review the content and assign it to one of the two categories based on these guidelines."""
sign_list = [w, x, y, z]
class SensSignatureW(dspy.Signature):
    """Classify an email message from a work inbox as containing sensitive personal information or not. Messages with sensitive personal information can be purely personal (unrelated to work). Additionally, they may contain sensitive personal information in a professional context, such as comments on work quality or feelings about employee treatment, while excluding discussions of company business or strategy. Assign the message to one of two categories: sensitive or not sensitive."""

    question = dspy.InputField(prefix="Message:")
    answer = dspy.OutputField(desc="", prefix="Message classification:")

class SensSignatureX(dspy.Signature):
    """Classify an email message from a work inbox as sensitive or not sensitive based on the presence of sensitive personal information. An email is sensitive if it includes personal information not directly related to work tasks, such as personal life details or issues. It's also sensitive if it contains personal opinions or feelings about work, like comments on work quality or sentiments about employee treatment. Note that routine mentions of names, addresses, or phone numbers in a work context, like scheduling meetings or assigning tasks, are not considered sensitive. Emails that discuss company business, strategies, or professional tasks without divulging personal or confidential information should be classified as not sensitive. Review the content carefully to determine the correct category based on these guidelines."""

    question = dspy.InputField(prefix="Message:")
    answer = dspy.OutputField(desc="", prefix="Message classification:")

class SensSignatureY(dspy.Signature):
    """Classify an email message from a work inbox as sensitive or not sensitive based on the presence of sensitive personal information. An email is sensitive if it includes personal information not directly related to work tasks, such as personal life details or issues. It's also sensitive if it contains personal opinions or feelings about work, like comments on work quality or sentiments about employee treatment. Note that routine mentions of names, addresses, or phone numbers in a work context, like scheduling meetings or assigning tasks, are not considered sensitive. Emails that discuss company business, strategies, or professional tasks without divulging personal information should be classified as not sensitive. Review the content carefully to determine the correct category based on these guidelines."""

    question = dspy.InputField(prefix="Message:")
    answer = dspy.OutputField(desc="", prefix="Message classification:")

class SensSignatureZ(dspy.Signature):
    """Classify an email message from a work inbox as sensitive or not sensitive based on the presence of sensitive personal information. An email should be categorized as sensitive if it includes personal information unrelated to work tasks, such as personal life details or issues. Additionally, classify an email as sensitive if it contains personal opinions or feelings related to work, like comments on work quality or sentiments about employee treatment. Note that emails discussing company business, strategies, or professional tasks, without personal information or opinions, should not be considered sensitive. If an email does not meet the criteria for being sensitive, categorise it as not sensitive. Carefully review the content and assign it to one of the two categories based on these guidelines."""

    question = dspy.InputField(prefix="Message:")
    answer = dspy.OutputField(desc="", prefix="Message classification:")

class SensSignatureitspers(dspy.Signature):
    """<s>[INST] Your task is to determine if the email message from a work email contains personal information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Does the message contain purely personal information or information that is personal a professional context? [/INST] """

    question = dspy.InputField(prefix="Message:")
    answer = dspy.OutputField(desc="", prefix="Message classification:")

class SensSignatureitspersend(dspy.Signature):
    """<s>[INST] Your task is to determine if the email message from a work email contains personal information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Does the message contain purely personal information or information that is personal a professional context? [/INST] """

    question = dspy.InputField(prefix="Message:")
    answer = dspy.OutputField(desc="", prefix="The text does")

class SensSignatureitspersclean(dspy.Signature):
    """Your task is to determine if the email message from a work email contains personal information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Does the message contain purely personal information or information that is personal a professional context?"""

    question = dspy.InputField(prefix="Message:")
    answer = dspy.OutputField(desc="", prefix="The text does")

class PromptNNEx(dspy.Module):
    def __init__(self, config, sig):
        super().__init__()

        self.signature = sig
        self.predictor = dspy.ChainOfThought(self.signature, activated=False) #, rationale_type=x)
        self.config = config

    def forward(self, question):
        result = self.predictor(question=question, **self.config)
        return dspy.Prediction(
            answer=result.answer,
        )


#sign_list2 = [SensSignatureW, SensSignatureX, SensSignatureY, SensSignatureZ]
#sign_list3 = ['W', 'X', 'Y', 'Z']

sign_list2 = [SensSignatureitspers, SensSignatureitspersend, SensSignatureitspersclean]
sign_list3 = ['itspers', 'ending', 'notags']
turbo = dspy.HFModel(model = "mistralai/Mistral-7B-Instruct-v0.2") #"meta-llama/Llama-2-7b-chat-hf")
dspy.settings.configure(lm=turbo)
for i in range(len(sign_list2)):
    mrs = main_experiment(PromptNNEx, sign_list2[i], 10000)
    write_responses_json(mrs, f'results/mist{sign_list3[i]}.json')
    #break

#mrs = main_experiment(PromptNN, 1)
#print(mrs)
#write_responses_json(mrs, 'results/mistchunks.json')

