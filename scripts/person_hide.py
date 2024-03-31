from dataset import load_sara
from preprocess_sara import full_preproc
import os
import pandas as pd
from transformers import AutoTokenizer
import spacy
nlp = spacy.load("en_core_web_sm")

s = load_sara()
m = 'mistralai/Mistral-7B-Instruct-v0.2'
t = AutoTokenizer.from_pretrained(m, use_fast=True)
anon_text = []
for d in s.text:
    doc = nlp(d)
    anonymized_text = d
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            anonymized_text = anonymized_text.replace(ent.text, "") #"<person>")
        #elif ent.label_ == "GPE":
        #    anonymized_text = anonymized_text.replace(ent.text, "<LOCATION>")

    anon_text.append(anonymized_text)

new_list = [{'doc_id':r.doc_id, 'text':anon_text[i], 'sensitivity':r.sensitivity} for i, r in s.iterrows()]
s = pd.DataFrame.from_dict(new_list)
print(s)

p = full_preproc(s, t)
print(p)