from dataset import load_sara
import pandas as pd
import spacy
from transformers import AutoTokenizer
import email
import gensim
import re
import numpy as np


def nameless_preproc(s, tokenizer, c_size=2048):

    def clean_names(data, replaced=''):
        nlp = spacy.load("en_core_web_sm")
        anon_text = []
        for d in data.text:
            doc = nlp(d)
            anonymized_text = d
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    anonymized_text = anonymized_text.replace(ent.text, replaced)
            anon_text.append(anonymized_text)
        #data = data.reset_index()
        new_list = [{'doc_id':r.doc_id, 'text':anon_text[i], 'sensitivity':r.sensitivity} for i, r in data.iterrows()]
        return pd.DataFrame.from_dict(new_list)

    def preprocess(e):
        message = email.message_from_string(e)
        clean = message.get_payload()
        clean = re.sub('\S*@\S*\s?', '', clean)
        clean = re.sub('\s+', ' ', clean)
        clean = re.sub("\'", "", clean)
        clean = gensim.utils.simple_preprocess(str(clean), deacc=True, min_len=1, max_len=100) 
        #clean = clean.lower()
        #clean = clean.translate(str.maketrans('','', string.punctuation))
        #clean = clean.translate(str.maketrans('','', "-_?"))
        return clean

    def remove_doubles(df):
        already_exists = []
        unique_df = []
        for i, s in enumerate(df.iterrows()):
            idd = s[1].doc_id
            text = s[1].text
            sensitivity = s[1].sensitivity
            if text in already_exists:
                continue
            already_exists.append(text)
            unique_df.append({'doc_id': idd, 'text':text, 'sensitivity':sensitivity})    
        return pd.DataFrame.from_dict(unique_df)

    def chunk(text, tokenizer, c_size):
        new_chunks = []
        tokens= tokenizer(text, return_tensors="pt")
        total_length = len(tokens.input_ids[0])
        avg_chunks = np.ceil(total_length / c_size)
        for i in range(int(avg_chunks)):
            chunk = tokens.input_ids[0][(i*c_size):((i+1)*c_size)]
            chunk = tokenizer.decode(chunk, skip_special_tokens=True)
            new_chunks.append(chunk)

        return new_chunks
        
    def chunk_large(df, place, tokenizer, c_size):
        place_docs = [dno[0] for dno in place]
        new_docs = []
        existing_texts = []
        for i, s in enumerate(df.iterrows()):
            ids = s[1].doc_id
            sens = s[1].sensitivity
            te = s[1].text

            if i not in place_docs:
                
                new_chunks = chunk(te, tokenizer, c_size)
                if len(new_chunks) == 1:
                    new_docs.append({'doc_id':ids, 'text':te, 'sensitivity':sens})
                    continue

                cut = 0
                for c in new_chunks:
                    new_docs.append({'doc_id':ids+'_'+str(cut), 'text':c, 'sensitivity':sens})
                    cut += 1
                continue        
                #new_docs.append({'doc_id':ids, 'text':te, 'sensitivity':sens})
                #continue
        return new_docs
    
    def remove_unnecessary(x):
        stop_words_extra = [' from ', ' e ', ' mail ', ' cc ', 'forwarded', ' by ', 'original ', ' message ', ' enron ', ' on ', ' pm ', ' am ']
        for w in stop_words_extra:
            x = re.sub(w, ' ', x)
        return x

    def main(s):
        s = clean_names(s)
        processed_emails = [preprocess(a) for a in s.text]
        ids = s.doc_id.tolist()
        sens = s.sensitivity.tolist()
        texts = []
        for i, text in enumerate(s.text):
            new_email = ' '.join(processed_emails[i])
            texts.append(new_email)

        new_dict = {'doc_id': ids, 'text': texts, 'sensitivity':sens}
        preproc_df = pd.DataFrame.from_dict(new_dict)
        preproc_df = remove_doubles(preproc_df)
        #places = get_replies(preproc_df)
        places = []
        preproc_df['text'] = preproc_df['text'].apply(lambda x: remove_unnecessary(x))
        new_docs = chunk_large(preproc_df, places, tokenizer, c_size)
        new_docs = pd.DataFrame.from_dict(new_docs)
        return new_docs

    return main(s)



def main():
    s = load_sara()
    m = 'mistralai/Mistral-7B-Instruct-v0.2'
    t = AutoTokenizer.from_pretrained(m, use_fast=True)
    p = nameless_preproc(s, t)
    print(p)
    stop_words_extra = [' from ', ' e ', ' mail ', ' cc ', 'forwarded', ' by ', 'original ', ' message ', ' enron ', ' on ', ' pm ', ' am ']
    for c in p.text:
        if c in stop_words_extra:
            print(22)
            break

#main()
