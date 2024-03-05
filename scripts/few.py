
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
from preprocess_sara import proccutit
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

def get_max_sims(s, key_vecs, samp=None):
    bp = 0
    max_sims = []
    max_sims_is = []
    if samp is None:
        samp_key_vecs = key_vecs
    else:
        sm = s.sample(n=samp, random_state=1)
        samp_key_vecs = {}
        for k in sm.doc_id:
            samp_key_vecs[k] = key_vecs.get(k)

    for k, embed in samp_key_vecs.items():
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
            seq_label = s[s['doc_id']==k].sensitivity.iloc[0]
            if seq_label == 0:
                non_sens_dict[i] = c
            elif seq_label == 1:
                sens_dict[i] = c

        if len(sims) != 1701:
            print(len(sims))

        #max_sims.append((non_sens_max_sim, sens_max_sim))
        #max_sims_is.append((non_sens_max_sim_i, sens_max_sim_i))
        l = [k for k, v in sorted(sens_dict.items(), key=lambda item: item[1], reverse=True)]
        m = [k for k, v in sorted(non_sens_dict.items(), key=lambda item: item[1], reverse=True)]

        max_sims_is.append((m,l))

        
        bp += 1
        if (bp % 100) == 0:
            print('Iteration', bp)
        if bp == samp:
            break
    
    return max_sims_is


def key_sim(s, max_sims_is, samp=None):
    key_to_sims = {}
    if samp == None:
        pass
    else:
        s = s.sample(n=samp, random_state=1)
    for i, k in enumerate(s.doc_id):
        if (len(max_sims_is)-1) < i:
            break
        key_to_sims[k] = max_sims_is[i]

    return key_to_sims


def get_sim_text(s, index, proc):
    sim = s.loc[index]
    id_sim = sim.doc_id
    proc = proccutit(s)
    filtered_df = proc[proc['doc_id'].apply(lambda x: (x.startswith(id_sim)))]
    if len(filtered_df) == 0:
        return ' '
    nonsen_few_prompt = filtered_df.iloc[0].text
    return nonsen_few_prompt

def get_sims(nonsen, sens, s, proc):
    #nonsen, sens = max_sims_is[i]
    nonsen_few_prompt = get_sim_text(s, nonsen, proc)
    sen_few_prompt = get_sim_text(s, sens, proc)
    return nonsen_few_prompt, sen_few_prompt


def get_key_to_sims(samp=None):
    s = load_sara()
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_fast=True)
    encode_map = encode_texts(s, tokenizer)
    max_sims_is = get_max_sims(s, encode_map, samp)
    key_to_sims = key_sim(s, max_sims_is, samp)
    return key_to_sims

def main():
    samp = 2000 # bigger so won't end
    start = time.time()
    s = load_sara()
    proc = proccutit(s)
    #s.head()
    #tokenizer, model = get_model_version('get_model', "mistralai/Mistral-7B-Instruct-v0.2", "main", "auto")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_fast=True)
    encode_map = encode_texts(s, tokenizer)
    max_sims_is = get_max_sims(s, encode_map, samp)
    key_to_sims = key_sim(s, max_sims_is)
    end = time.time()
    duration = end-start
    print(duration)
    #print(key_to_sims)


#main()