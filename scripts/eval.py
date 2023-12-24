import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import numpy as np

def get_metric_dict(method, labels, preds):
    acc = accuracy_score(labels, preds)
    bac = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    
    metric_dict = {'accuracy': acc, 'f1_score': f1, 'balanced accuracy': bac}
    metrics_data[method] = metric_dict


prompts = ['base_prompt_template', 'context_prompt_template']
metrics_data = {}
for prompt in prompts:
    ground_truths = np.loadtxt(prompt+'_truth_labs.txt')
    preds = np.loadtxt(prompt+'_preds.txt')
    get_metric_dict('Llama-2', ground_truths, preds)
    
#print(metrics_data)
metrics_df = pd.DataFrame.from_dict(metrics_data, orient='index')
print(metrics_df)
