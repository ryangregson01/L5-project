import ir_datasets
import email
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import gc
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import time
import numpy as np

def get_sara():
    return ir_datasets.load('sara')

def sara_docs():
    for doc in sara_dataset.docs_iter():
        doc_id = doc.doc_id
        text = doc.text
        sens = doc.sensitivity
        text_method = doc.default_text()
        
        e = email.message_from_string(text_method)
        pl = e.get_payload()

        print(doc_id, sens)
        break

def dataset_to_df():
    doc_ids = []
    doc_text = []
    doc_sens = []
    for doc in sara_dataset.docs_iter():
        doc_ids.append(doc.doc_id)
        doc_text.append(doc.text)
        doc_sens.append(doc.sensitivity)

    sara_dict = {'doc_id':doc_ids, 'text':doc_text, 'sensitivity':doc_sens}
    df = pd.DataFrame.from_dict(sara_dict)
    return df

def base_prompt_template(document):
  return f"""[INST]
  Which one of the attributes "sensitive", "non-sensitive" classifies the Text.
  The Text is delimited by triple backticks. You must answer after 'Attribute:'[/INST]

  Text: ```{document}```.

  Attribute:
  """

def context_prompt_template(document):
  return f"""[INST]
  Which one of the attributes “sensitive”, “non-sensitive” classifies the Text.
  When classifying the Text only consider personal information that is sensitive. 
  The Text is delimited by triple backticks. You must answer after 'Attribute:'[/INST]
  
  Text: ```{document}```.

  Note: You must choose the attribute "sensitive" or "non-sensitive" to classify the Text and output only the attribute after 'Attirbute:'.
  Attribute: 
  """


def get_prompt_template(document, prompt):
   return prompt(document)

def llm_inference(document, prompt, model):
  inputs = tokenizer(get_prompt_template(document, prompt), return_tensors='pt')
  generation_config = GenerationConfig(
    # Unable to set temperature to 0 - https://github.com/facebookresearch/llama/issues/687 - use do_sample=False for greedy decoding
    do_sample=False,
    max_new_tokens=20,
  )
  use_cuda = True
  device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
  
  output = model.generate(inputs=inputs.input_ids.to(device), attention_mask=inputs.attention_mask.to(device), generation_config=generation_config)
  return tokenizer.decode(output[0], skip_special_tokens=True)

def display_gen_text(output):
  return output[-20:]

def prompt_to_reply(d, p):
  response = llm_inference(d, p, model)
  return display_gen_text(response)

# String matching on model response
def post_process_classification(classification, ground_truth):
    if 'non-sensitive' in classification.lower():
        if ground_truth == 0:
            return 'TN', 0
        else:
            return 'FN', 0

    elif 'sensitive' in classification.lower() and 'non-sensitive' not in classification.lower():
        if ground_truth == 1:
            return 'TP', 1
        else:
            return 'FP', 1

    else:
        # Further processing required
        return classification, None
        further_processing_required[sample[1].doc_id] = classification


def clear_memory():
    # Prevents cuda out of memory
    torch.cuda.empty_cache()
    gc.collect()


# Dataset - dataframe, prompt_strategy - prompt function name
def llm_experiment(dataset, prompt_strategy=base_prompt_template):
    predictions = {
        'TP' : 0, # Sensitive
        'FP' : 0, # Non-sensitive document classified as sensitive
        'TN' : 0, # Non-sensitive
        'FN' : 0,
    }
    # Model output is not an expected sensitivity attribute
    further_processing_required = {}
    # All model output
    model_responses = {}
    
    #total_samples = 1

    scikit_true = []
    scikit_pred = []

    for sample in dataset.iterrows():
        sample_text = sample[1].text
        ground_truth = sample[1].sensitivity

        # To replace with appropriate pre-processing
        if len(sample_text) > 12000:
            continue
        
        classification = prompt_to_reply(sample_text, prompt_strategy)
        model_responses[sample[1].doc_id] = classification

        quadrant, pred = post_process_classification(classification, ground_truth)
        if pred == None:
            further_processing_required[sample[1].doc_id] = quadrant
            continue

        predictions[quadrant] = predictions.get(quadrant) + 1
        scikit_true.append(ground_truth)
        scikit_pred.append(pred)

        clear_memory()

        #total_samples -= 1
        #if total_samples == 0:
        #    break

    return predictions, further_processing_required, model_responses, scikit_true, scikit_pred


# Main
sara_dataset = get_sara()
sara_df = dataset_to_df()

testing_sample = sara_df.sample(frac=0.2, random_state=1)
value_counts = testing_sample.sensitivity.value_counts()
doc_lengths = testing_sample.text.str.len()
temp_testing_sample = testing_sample[doc_lengths < 12000]
testing_sample = temp_testing_sample
sampled_indices = testing_sample.index
training_data = sara_df.drop(sampled_indices)

clear_memory()
clear_memory()
clear_memory()

access_token = l2_token
model_name =  "meta-llama/Llama-2-7b-chat-hf" #"TheBloke/Llama-2-7B-Chat-GPTQ" #"meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", use_auth_token=access_token, cache_dir=my_cache)
print(model_name)

prompts = [base_prompt_template, context_prompt_template]

for prompt in prompts:
    prompt_str = str(prompt).split()[1] + '_'
    start = time.time()
    predictions, further_processing_required, model_responses, scikit_true, scikit_pred = llm_experiment(testing_sample, prompt)
    end = time.time()
    
    duration = end-start

    truth_labs = np.array(scikit_true)
    preds = np.array(scikit_pred)

    np.savetxt(prompt_str+'truth_labs.txt', truth_labs)
    np.savetxt(prompt_str+'preds.txt', preds)

    f = open(prompt_str+"duration.txt", "w")
    f.write(str(duration))
    f.close()

