from dataset import load_sara
from preprocess_sara import full_preproc
from models import get_model_version
import torch
from final_prompts import *


def prompt_hint(d, hint):
    return f"""[INST] <<SYS>> 
Your task is to determine if the following email message contains sensitive personal information that should be protected, or is a non-personal email.

You are reviewing emails from the company Enron that are related to the California energy crisis, trading discrepancies, and other messages occurring near the end of the company's life.

Personal messages may be purely personal and do not include any relation to work being done. Personal messages may also be related to work and contain sentiment towards colleagues (e.g., it was good working with you). Non-personal messages discuss topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support), employment arrangements (job seeking, hiring, recommendations), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "personal", or "non-personal" describes the following message? The message should be classified as "personal" if any part of the message contains sensitive personal information. The hint should help you answer.
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {d}. 

Hint: {hint}
[/INST] 

Answer:"""

sara_df = load_sara()


tokenizer, model = get_model_version('get_model', 'TheBloke/Mistral-7B-Instruct-v0.2-AWQ', 'main', 'auto')


#samp = sara_df.sample(n=n, random_state=1)
processed_sara_df = full_preproc(sara_df, tokenizer)

document = '[INST] Explain what: "I heard you were a big hit" means. [/INST]'
#'This looks like a run-of-the-mill conference except that the guest list is fairly senior.'

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 

encodeds = tokenizer(document, return_tensors="pt", padding=True)
model_inputs = encodeds.to('cuda')
with torch.no_grad():
    generated_ids = model.generate(inputs=model_inputs.input_ids, 
        attention_mask=model_inputs.attention_mask, 
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        max_new_tokens=150, # 150
    )
decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
x = decoded[0]
x = x[x.find('[/INST]')+8:]
print(x)

hint = x
doc = processed_sara_df[processed_sara_df.doc_id=='175662'].iloc[0].text
document = prompt_hint(doc, hint)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 

encodeds = tokenizer(document, return_tensors="pt", padding=True)
model_inputs = encodeds.to('cuda')
with torch.no_grad():
    generated_ids = model.generate(inputs=model_inputs.input_ids, 
        attention_mask=model_inputs.attention_mask, 
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        max_new_tokens=150, # 150
    )
decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(decoded[0])
