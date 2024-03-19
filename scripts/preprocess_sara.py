import numpy as np
import pandas as pd
import re
import email
import gensim

def full_preproc(s, tokenizer):

    def preprocess(e):
        message = email.message_from_string(e)
        clean = message.get_payload()
        clean = re.sub('\S*@\S*\s?', '', clean)
        clean = re.sub('\s+', ' ', clean)
        clean = re.sub("\'", "", clean)
        clean = gensim.utils.simple_preprocess(str(clean), deacc=True, min_len=1, max_len=100) 
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

    def get_replies(df):
        place = []
        for i, tex in enumerate(df.text):
            words = tex.split()
            for j, word in enumerate(words):

                if 'forwarded' == word:
                    if words[j+1] == 'by':
                        place.append((i, j))
                        continue

                if 'original' == word:
                    if words[j+1] == 'message':
                        place.append((i, j))
                        continue

        return place
    '''
    def chunk(text, c_size=9000):
        new_chunks = []
        total_length = len(text)
        avg_chunks = np.ceil(total_length / c_size)
        words = text.split()
        no_words = len(words)
        words_per_chunk = int(np.ceil(no_words / avg_chunks))
        #print(words_per_chunk)
        for i in range(int(avg_chunks)):
            chunk = ' '.join(words[(i*words_per_chunk):((i+1)*words_per_chunk)])
            new_chunks.append(chunk)

        return new_chunks
    '''
    def chunk(text, tokenizer, c_size=2048):
        new_chunks = []
        tokens= tokenizer(text, return_tensors="pt")
        total_length = len(tokens.input_ids[0])
        avg_chunks = np.ceil(total_length / c_size)
        for i in range(int(avg_chunks)):
            chunk = tokens.input_ids[0][(i*c_size):((i+1)*c_size)]
            chunk = tokenizer.decode(chunk, skip_special_tokens=True)
            new_chunks.append(chunk)

        return new_chunks
        
    def chunk_large(df, place, tokenizer):
        place_docs = [dno[0] for dno in place]
        new_docs = []
        existing_texts = []
        for i, s in enumerate(df.iterrows()):
            ids = s[1].doc_id
            sens = s[1].sensitivity
            te = s[1].text

            if i not in place_docs:
                
                new_chunks = chunk(te, tokenizer)
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

            words = te.split()
            cut_pos_init = 0
            cut = 0
            for pair in place:
                if pair[0] == i:
                    cut_pos = pair[1]
                    seg = words[cut_pos_init:cut_pos]
                    cut_pos_init = cut_pos
                    text_join = ' '.join(seg)
                    
                    new_chunks = chunk(text_join, tokenizer)
                    if len(new_chunks) == 1:
                        x = {'doc_id':ids+'_'+str(cut), 'text':text_join, 'sensitivity':sens}
                        cut += 1
                        new_docs.append(x)

                    else:
                        for c in new_chunks:
                            new_docs.append({'doc_id':ids+'_'+str(cut), 'text':c, 'sensitivity':sens})
                            cut += 1
                    
                    #x = {'doc_id':ids+'_'+str(cut), 'text':text_join, 'sensitivity':sens}
                    #cut += 1
                    #new_docs.append(x)


            seg = words[cut_pos_init:]
            text_join = ' '.join(seg)
            
            new_chunks = chunk(text_join, tokenizer)
            if len(new_chunks) == 1:
                x = {'doc_id':ids+'_'+str(cut), 'text':text_join, 'sensitivity':sens}
                cut += 1
                new_docs.append(x)
            else:
                for c in new_chunks:
                    new_docs.append({'doc_id':ids+'_'+str(cut), 'text':c, 'sensitivity':sens})
                    cut += 1
            
            #new_docs.append({'doc_id':ids+'_'+str(cut), 'text':c, 'sensitivity':sens})
        return new_docs

    def main(s):
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
        places = get_replies(preproc_df)
        new_docs = chunk_large(preproc_df, places, tokenizer)
        new_docs = pd.DataFrame.from_dict(new_docs)
        return new_docs

    return main(s)



def proccutit(dataset):
    def pre_dict(d):
        preproc = {}
        clean = []
        clean_docus = {}
        k_to_sens = {}
        for s in d.iterrows():
            k = s[1].doc_id
            v = s[1].text
            sens = s[1].sensitivity
            v = v.split('\r\n\r\n')[1]

            multi_split = s[1].text.split('\r\n\r\n')
            if len(multi_split) > 2:
                reconnect = multi_split[1:]
                v = '\n\n'.join(reconnect)

            cleaned_text = v
            cleaned_text = re.sub(r'=\r\n', '\n', cleaned_text)
            cleaned_text = re.sub(r'\r\n', '\n', cleaned_text)
            cleaned_text = re.sub(r'\n>\n', '\n\n', cleaned_text)
            # Or replace with ''
            # Put space in #cleaned_text = re.sub(r'\n> ', '\n', cleaned_text)
            cleaned_text = re.sub(r'\n> ', '\n ', cleaned_text)                         
            cleaned_text = re.sub(r'\n\s+\n', '\n\n', cleaned_text)
            cleaned_text = re.sub(r'\?{2,}', '', cleaned_text)
            cleaned_text = re.sub(r'\n\?+', '', cleaned_text)
            cleaned_text = re.sub(r' \?+', '', cleaned_text)
            cleaned_text = re.sub(r'\?{2,}', '', cleaned_text)
            cleaned_text = re.sub(r'=20', ' ', cleaned_text)
            cleaned_text = re.sub(r'=09', ' ', cleaned_text)
            clean.append(cleaned_text)

            cutit = cleaned_text.split('\n\n')
            sep = []
            clean_docus[k] = []
            keywords = {'To:', 'cc:', 'Sent:', 'From:', 'mailto'}
            for cut in cutit:
                if not any(keyword in cut for keyword in keywords):
                    sep.append(cut)
                else:
                    clean_docus[k].append(sep)
                    sep = []
            clean_docus[k].append(sep)
            k_to_sens[k] = sens

        preproc = {}
        ids = []
        texts = []
        sens_vals = []
        for k, v in clean_docus.items():
            if len(v) == 1:
                para = v[0][0]
                full_m = ''.join(para.split('\n'))
                if len(full_m) < 15:
                    continue
                #preproc[k] = full_m
                ids.append(k)
                texts.append(full_m)
                sens_vals.append(k_to_sens.get(k))
            else:
                for i, m in enumerate(v):
                    connected_m = []

                    # Catch footer
                    if len(m) > 1:
                        if len(m[-1]) < 4:
                            m = m[:-1]
                        
                        if '@' in m[-1] or 'fax:' in m[-1].lower() or 'Sent by:' in m[-1]:
                            m = m[:-1]
                        else:
                            pass
                            #print(m[-1])

                    for para in m:
                        message = ''.join(para.split('\n'))
                        connected_m.append(message)

                    full_m = '\n'.join(connected_m)
                    if len(full_m) < 15:
                        continue
                    id_part = k + '_' + str(i)
                    #preproc[id_part] = full_m
                    ids.append(id_part)
                    texts.append(full_m)
                    sens_vals.append(k_to_sens.get(k))

        preproc['doc_id'] = ids
        preproc['text'] = texts
        preproc['sensitivity'] = sens_vals
        return preproc

    def cutit(datadict):
        pre = pre_dict(datadict)
        preproc_df = pd.DataFrame.from_dict(pre)
        return preproc_df

    return cutit(dataset)
