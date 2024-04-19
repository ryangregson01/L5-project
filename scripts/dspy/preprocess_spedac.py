import pandas as pd
import csv

def load_proc():

    f = open("spedac-test.txt", "r")
    l = f.read()
    l = l.split('\n\n')
    s = l[2:]

    ids = []
    texts = []
    sens = []
    for i_id, v in enumerate(s):
        ids.append(i_id)
        v_split = v.split('\n')
        text = v_split[0][6:]

        real_word = []
        sens_attr = 0
        for word in v_split[1:]:
            row_split = word.split('\t')
            real_word.append(row_split[2])

            if row_split[3][0] == 'S':
                sens_attr = 1

        join_r_w = ' '.join(real_word)
        join_r_w = join_r_w.strip('"')
        join_r_w = join_r_w.strip(' ')
        texts.append(join_r_w)
        sens.append(sens_attr)

    full_dict = {'doc_id': ids, 'text': texts, 'sensitivity': sens}

    x = pd.DataFrame.from_dict(full_dict)
    return x

y = load_proc()
print(y.head())
