
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import gc
from config import *
import re
import numpy as np
import sys

from dataset import load_sara
from models import get_model_version
from preprocess_sara import proccutit, full_preproc
from sklearn.metrics.pairwise import cosine_similarity
import time

# Get all embeddings
def encode_texts(s, tokenizer):
    vecs = {}
    for samp in s.iterrows():
        idd = samp[1].doc_id
        text = samp[1].text
        encoded_text = tokenizer(text, return_tensors="pt")
        vector_rep = encoded_text.input_ids[0]
        vecs[idd] = vector_rep
    return vecs


# Get an array of most similar documents where each index is ordered keys
# and follow (nonsens, sens).
def pad_token_sequences(seq1, seq2, pad_token_id=0):
    max_len = max(len(seq1), len(seq2))
    seq1_padded = seq1 + [pad_token_id] * (max_len - len(seq1))
    seq2_padded = seq2 + [pad_token_id] * (max_len - len(seq2))
    return seq1_padded, seq2_padded

def get_max_sims(fproc, sproc, key_vecs):
    bp = 0
    max_sims = []
    max_sims_is = []

    for k, embed in key_vecs.items():
        sims = []
        sens_max_sim = 0
        sens_max_sim_i = 0
        non_sens_max_sim = 0
        non_sens_max_sim_i = 0
        non_sens_dict = {}
        sens_dict = {}
        token_sequence_1 = embed
        token_sequence_1 = token_sequence_1.tolist()
        for i, k in enumerate(key_vecs.keys()):
            seq = key_vecs[k]
            if embed is seq:
                continue

            seq = seq.tolist()
            seq1_padded, seq_padded = pad_token_sequences(token_sequence_1, seq)
            # Now seq1_padded and seq_padded are ready for comparison
            c = cosine_similarity([seq1_padded], [seq_padded])
            sims.append(c)

            c = c.item()
            seq_label = fproc[fproc['doc_id']==k].sensitivity.iloc[0]
            if seq_label == 0:
                non_sens_dict[i] = c
            elif seq_label == 1:
                sens_dict[i] = c

        if len(sims) != (len(fproc) - 1):
            print(len(sims))

        #max_sims.append((non_sens_max_sim, sens_max_sim))
        #max_sims_is.append((non_sens_max_sim_i, sens_max_sim_i))
        l = [k for k, v in sorted(sens_dict.items(), key=lambda item: item[1], reverse=True)]
        m = [k for k, v in sorted(non_sens_dict.items(), key=lambda item: item[1], reverse=True)]

        combined_dict = {**sens_dict, **non_sens_dict}
        #print(len(combined_dict))
        c = [k for k, v in sorted(combined_dict.items(), key=lambda item: item[1], reverse=True)]
        max_sims_is.append(c)

        #max_sims_is.append((m,l))

        
        bp += 1
        if (bp % 100) == 0:
            print('Iteration', bp)
        if bp == len(sproc):
            break
    
    return max_sims_is


def key_sim(sproc, max_sims_is):
    key_to_sims = {}
    for i, k in enumerate(sproc.doc_id):
        if (len(max_sims_is)-1) < i:
            break
        key_to_sims[k] = max_sims_is[i]

    return key_to_sims


def get_sim_text(fproc, index):
    sim = fproc.loc[index].text
    sim_label = fproc.loc[index].sensitivity
    #print(sim)
    return sim, sim_label
    id_sim = sim.doc_id
    #filtered_df = sproc[sproc['doc_id'].apply(lambda x: (x.startswith(id_sim)))]
    if len(filtered_df) == 0:
        return ' '
    nonsen_few_prompt = filtered_df.iloc[0].text
    return nonsen_few_prompt

def get_sims(nonsen, sens, fproc, doc_len):
    #nonsen, sens = max_sims_is[i]
    shot_length = (9500 - doc_len) / 2
    shot_nonsen = ''
    shot_sen = ''
    for val in nonsen:
        text = get_sim_text(fproc, val)
        if len(text) <= shot_length:
            shot_nonsen = text
            break

    #print(shot_nonsen)

    for val in sens:
        text = get_sim_text(fproc, val)
        if len(text) <= shot_length:
            shot_sen = text
            break

    #print(shot_sen)
    #nonsen_few_prompt = get_sim_text(fproc, nonsen)
    #sen_few_prompt = get_sim_text(fproc, sens)
    return shot_nonsen, shot_sen


def new_get_sims(sim_docs, fproc, doc_len):
    pass
    shot_length = (9500 - doc_len) / 2
    shot_one = ''
    label_one = -1
    shot_two = ''
    label_two = -1
    for val in sim_docs:
        text, label = get_sim_text(fproc, val)
        if len(text) <= shot_length:
            if shot_one == '':
                shot_one = text
                label_one = label
            else:
                shot_two = text
                label_two = label
                break

    return shot_one, label_one, shot_two, label_two

def sing_get_sims(sim_docs, fproc, doc_len):
    shot_length = 9500 - doc_len
    lookup = {0: 'non-personal', 1:'personal'}
    for val in sim_docs:
        text, label = get_sim_text(fproc, val)
        if len(text) <= shot_length:
            return text, lookup.get(label)
        
    return " ", "non-personal"

def get_max_sims_call(fproc, sproc, tokenizer):
    encode_map = encode_texts(fproc, tokenizer)
    max_sims_is = get_max_sims(fproc, sproc, encode_map)
    key_to_sims = key_sim(sproc, max_sims_is)
    #print(key_to_sims)
    return key_to_sims


def get_key_to_sims(full_proc, sproc, tokenizer):
    #samp=2
    #s = load_sara()
    #tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_fast=True)
    #full_proc = full_preproc(s, tokenizer)
    #samp_proc = s.sample(n=samp, random_state=1)
    key_sims = get_max_sims_call(full_proc, sproc, tokenizer)
    return key_sims


def main():
    samp = 2 #2000 # bigger so won't end
    start = time.time()
    s = load_sara()
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_fast=True)
    proc = full_preproc(s, tokenizer)
    #tokenizer, model = get_model_version('get_model', "mistralai/Mistral-7B-Instruct-v0.2", "main", "auto")
    encode_map = encode_texts(s, tokenizer)
    max_sims_is = get_max_sims(s, encode_map, samp)
    key_to_sims = key_sim(s, max_sims_is)
    end = time.time()
    duration = end-start
    print(duration)
    #print(key_to_sims)


def main():
    samp = 2 #2000 # bigger so won't end
    s = load_sara()
    #samp = s.sample(n=samp, random_state=1)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_fast=True)
    full_proc = full_preproc(s, tokenizer)
    samp_proc = full_proc.sample(n=samp, random_state=1)
    key_sims = get_key_to_sims(full_proc, samp_proc, tokenizer)
    #print(key_sims)
    doc = samp_proc[samp_proc.doc_id == '173146'].text.iloc[0] 
    print(doc)
    l = key_sims.get('54580')
    print(l[0])
    #get_sim_text(full_proc, l[1][0], samp_proc)

    #get_sims(l[0], l[1], full_proc, len(doc))
    x = new_get_sims(l, full_proc, len(doc))
    print(x)
    

#main()

