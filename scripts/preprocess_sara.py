import numpy as np
import pandas as pd
import re

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
