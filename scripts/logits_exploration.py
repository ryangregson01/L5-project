import time
import numpy as np
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import sys
sys.path.append("../")
from models import get_model_version
#from pipeline_config import model_map
import torch

model_map = {'l27b-meta': ['get_l2', 'meta-llama/Llama-2-7b-chat-hf', 'main'], # 2048 max tokens doc
            'l213b-8bit': ['get_l2', 'TheBloke/Llama-2-13B-chat-GPTQ', 'gptq-8bit-64g-actorder_True'], 
            'l270b-4bit': ['get_l2', 'TheBloke/Llama-2-70B-chat-GPTQ', 'main'],
            'mist7b-mist': ['get_model', 'mistralai/Mistral-7B-Instruct-v0.2', 'main'],
            'mixt-4bit': ['get_model', 'TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ', 'main'],
            'flan-large': ['get_flan', 'google/flan-t5-large', 'main'], # 320 max tokens doc
            'l2-nochat': ['get_l2', 'meta-llama/Llama-2-7b-hf', 'main'],
            'mist-awq': ['get_model', 'TheBloke/Mistral-7B-Instruct-v0.2-AWQ', 'main'],
            'l2bnb': ['get_l2_bits', 'meta-llama/Llama-2-13b-hf', 'main'],
            }


m, v, r = model_map.get('mist7b-mist')
d = 'auto'
tokenizer, model = get_model_version(m, v, r, d)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


input_text = """[INST] <<SYS>> 
Your task is to determine if the following email message contains sensitive personal information that should be protected, or is a non-personal email.

You are reviewing emails from the company Enron that are related to the California energy crisis, trading discrepancies, and other messages occurring near the end of the company's life.

Personal messages messages may be purely personal and do not include any relation to work being done. Personal messages that are related to work will contain comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages discuss topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support), employment arrangements (job seeking, hiring, recommendations), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "personal", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: though i had a somewhat different notion when i initially raised the idea of co sponsorship i agree with lees observations and think that we should proceed the way he suggests on pm to cc subject fw possible co sponsorships all lee asked me to forward this im still awaiting additional suggestions from anyone on speakers i guess lees e mail changes things if the business school wants to go forward with a conference anyway then it may be a bad idea to have a separate one jeff has said he likes the idea of coordinating bill and allen what do you think i will chime in that carl shapiro is very much a big wig as a former chief economist at doj as for the frank wolak suggestion frank is a stanford economist who is an outstanding analyst and has published probably more than anyone else on electricity market design performance regarding the uk australia and california he speaks a mile a minute though and his understanding of policy and politics is a bit naive i should note that icf will not be able to contribute i heard from michael berg this morning so he will not be participating in our discussions either my opinion is lets do whatever is best for the school one positive outcome of this would be stronger relationships with some of the universitys top notch economic policy faculty overshadowing is possible lee does dean nacht have a view on joint sponsorship rob original message from lee s friedman sent tuesday august pm to rob gramlich subject possible co sponsorships rob id send this to the whole group but i am at a different computer today and dont have all the email addresses perhaps you can forward this i just received a phone call from carl shapiro he began by saying that he and several people from the business school severin borenstein george cluff are planning an electricity deregulation mini conference that sounds exactly like ours and wanted to check so that we dont step on each others toes and perhaps can do it together they even had october in mind for their timing we are further along then they are however my first response to him was that because our event is alumni initiated i am not sure that they would want this to be other than a gspp event by the end of our conversation we were discussing gspp co sponsorship with two other campus units iber and uei neither are schools carl is director of the institute for business and economics research a campus wide organized research unit and rich gilbert assists in this uei is severins group the university wide energy research institute carl suggested that they could help with administration and perhaps some modest support if we do this together carl himself is on the market surveillance committee of the iso and i think would hope to have some speaking role he also mentioned frank wolak of stanford as a speaker i think it would be good to try and work out this co sponsorship it would mean allowing some of them carl and severin into our planning group there connections are probably very valuable to us and they really are on the same wavelength the alternative of gspp going it alone after this initiative seems to me to be bad feelings and crossed wires that would be no good to anyone reactions lee. 
[/INST] 

Answer: ["""


input_ids = tokenizer.encode(input_text, return_tensors='pt')
input_ids = input_ids.to('cuda')
outputs = model.generate(input_ids, max_new_tokens=6, output_scores=True, return_dict_in_generate=True)
tokens_of_interest = ['non', 'personal']
#token_ids_of_interest = {tokenizer.encode(token, add_special_tokens=False)[0]: token for token in tokens_of_interest}
#probabilities_of_interest = {token: [] for token in tokens_of_interest}
token_probabilities = []
generated_tokens = []
token_ids_of_interest = {4581:'non', 28545:'personal'}
probabilities_of_interest = {token: [] for token in tokens_of_interest}

# Process each token generated
for logits in outputs.scores:
    probs = torch.softmax(logits, dim=-1)
    for token_id, token in token_ids_of_interest.items():
        # Get the probability of the token of interest at each generation step
        token_probability = probs[0, token_id].item()
        probabilities_of_interest[token].append(token_probability)

    next_token_id = torch.argmax(probs, dim=-1).item()  # Most likely next token
    generated_tokens.append(next_token_id)
    token_probability = probs[0, next_token_id].item()
    token_probabilities.append(token_probability)

print(probabilities_of_interest)
k1 = probabilities_of_interest.get('non')
k2 = probabilities_of_interest.get('personal')
print(k1, k2)
if sum(k1) > 0.5:
    print(sum(k1))

'''
decoded_tokens = [tokenizer.decode([tok]) for tok in generated_tokens]
#print(decoded_tokens)
token_to_probability = list(zip(decoded_tokens, token_probabilities))
print(token_to_probability)

decoded_output = ' '.join(decoded_tokens)
print(decoded_output)
'''