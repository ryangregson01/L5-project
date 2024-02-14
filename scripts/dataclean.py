from dataset import load_sara
import numpy as np
import pandas as pd
import re


def proccutit(dataset):
    def get_separate_messages(message):
        #payload = message.split('\r\n\r\n')[1]
        separate_messages = []
        return separate_messages

    def pre_dict(d):
        preproc = {}
        clean = []
        clean_docus = {}
        for k,v in d.items():
            #print(v)
            #sep = get_separate_messages(v)
            cleaned_text = v
            cleaned_text = re.sub(r'\n>\n', '\n\n', cleaned_text)
            # Or replace with ''
            # Put space in
            #cleaned_text = re.sub(r'\n> ', '\n', cleaned_text)
            cleaned_text = re.sub(r'\n> ', '\n ', cleaned_text)                         
            cleaned_text = re.sub(r'\n\s+\n', '\n\n', cleaned_text)
            cleaned_text = re.sub(r'\?{2,}', '', cleaned_text)
            cleaned_text = re.sub(r'\n\?+', '', cleaned_text)
            cleaned_text = re.sub(r' \?+', '', cleaned_text)
            cleaned_text = re.sub(r'\?{2,}', '', cleaned_text)
            cleaned_text = re.sub(r'=20', ' ', cleaned_text)
            cleaned_text = re.sub(r'=09', ' ', cleaned_text)
            cleaned_text = re.sub(r'=\r\n', '\n', cleaned_text)
            clean.append(cleaned_text)

            cutit = cleaned_text.split('\n\n')
            sep = []
            clean_docus[k] = []
            keywords = {'To:', 'cc:', 'Sent:', 'From:'}
            for cut in cutit:
                if not any(keyword in cut for keyword in keywords):
                    sep.append(cut)
                else:
                    clean_docus[k].append(sep)
                    sep = []
                
            clean_docus[k].append(sep)


        preproc = {}
        for k, v in clean_docus.items():
            if len(v) == 1:
                para = v[0][0]
                full_m = ''.join(para.split('\n'))
                preproc[k] = full_m
            else:
                for i, m in enumerate(v):
                    connected_m = []
                    for para in m:
                        message = ''.join(para.split('\n'))
                        connected_m.append(message)

                    full_m = '\n'.join(connected_m)
                    id_part = k + '_' + str(i)
                    preproc[id_part] = full_m

        #print(preproc)
        return preproc
        for l in clean:
            # Check = gives \r\n
            if '=' in l:
                print(l)
                #decoded_body = quopri.decodestring(l)
                #print(decoded_body)
                #break
        return preproc

    def cutit(datadict):
        pre = pre_dict(datadict)
        return pre

    return cutit(dataset)


'''
sara = get_sara()
store = {}
full = {}
j = 0
for doc in sara.docs_iter():
    full[doc.doc_id] = doc.text
    d = doc.text.split('\r\n\r\n')[1]
    store[doc.doc_id] = d
    j += 1
    #if j == 4:
    #    break

#print(full)
#print(store)
x = proccutit(store)

print(len(x.keys()))
'''

'''
# Used when proccutit returns a dataframe
sara_df = load_sara()
samp = sara_df #.sample(n=2, random_state=1)
x = proccutit(samp)

#print(x.text.iloc[5])

print(len(x.doc_id))

for v in x.text:
    print(v)
'''