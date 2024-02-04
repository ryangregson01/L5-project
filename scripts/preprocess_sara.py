import email
import pandas as pd
import re

def proc3(dataset):
    def get_orig(message, find_orig_message):
        #line_start = message.rfind('\n', 0, find_orig_message)
        line_start = find_orig_message
        base_message = message[:line_start]
        line_end = find_orig_message + 5 #message.find('\n', find_orig_message)
        orig_message = message[line_end:]

        if 'Subject' in orig_message:

            over_header = orig_message.find('Subject:')
            orig_message = orig_message[over_header:]

        return base_message, orig_message

    def get_separate_messages(message):
        payload = message.split('\r\n\r\n')[1]
        separate_messages = []

        find_fwd_message = payload.find('- Forw')
        find_orig_message = payload.find('-----Original Message-')
        
        n = payload
        while (find_fwd_message > 0 or find_orig_message > 0):
            if (find_fwd_message > 0 and find_orig_message > 0):
                if find_fwd_message < find_orig_message:
                    b, n = get_orig(n, find_fwd_message)
                else:
                    b, n = get_orig(n, find_orig_message)
            elif (find_fwd_message > 0 and find_orig_message < 0):
                b, n = get_orig(n, find_fwd_message)
            else:
                b, n = get_orig(n, find_orig_message)
            
            find_fwd_message = n.find('- Forw')
            find_orig_message = n.find('-----Original Message-')

            b = re.sub(r'\s+', ' ', b)
            separate_messages.append(b)

        n = re.sub(r'\s+', ' ', n)
        if 'Subject' in n:
            over_header = n.find('Subject:')
            n = n[over_header:]
        separate_messages.append(n)

        return separate_messages

    def preprocessing_dataframe(testing_sample):
        ids = []
        texts = []
        sens = []
        preproc = {}

        for s in testing_sample.iterrows():
            separate_messages = get_separate_messages(s[1].text)

            if len(separate_messages) == 1:
                ids.append(s[1].doc_id)
                texts.append(separate_messages[0])
                sens.append(s[1].sensitivity)
            else:
                for i, m in enumerate(separate_messages):
                    id_part = s[1].doc_id + '_' + str(i)
                    ids.append(id_part)
                    texts.append(m)
                    sens.append(s[1].sensitivity)

        preproc['doc_id'] = ids
        preproc['text'] = texts
        preproc['sensitivity'] = sens
        return preproc

    def get_preprocessed_sara_p3(dataset):
        preproc = preprocessing_dataframe(dataset)
        preproc_df = pd.DataFrame.from_dict(preproc)
        return preproc_df

    return get_preprocessed_sara_p3(dataset)