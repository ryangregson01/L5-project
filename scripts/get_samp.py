from dataset import load_sara
from preprocess_sara import full_preproc
from config import *
from final_prompts import *

s = load_sara()
p = (full_preproc(s, 'mist-noreply'))
print(p)

x = all_cats(p[p.doc_id=='173153'].text.iloc[0])

print(x)