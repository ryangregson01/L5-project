import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

def get_metric_dict(method, labels, preds):
    acc = accuracy_score(labels, preds)
    bac = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    
    metric_dict = {'accuracy': acc, 'f1_score': f1, 'balanced accuracy': bac}
    metrics_data[method] = metric_dict


def evaluation_summary(description, true_labels, predictions):
    target_classes = ['Non-sensitive (0)', 'Sensitive (1)']
    print("Evaluation for: " + description)
    print(classification_report(true_labels, predictions, digits=3, zero_division=0, target_names=target_classes))
    print('\n\nConfusion matrix:')
    confusionMatrix = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=target_classes)
    disp.plot()
    plt.show()


prompts = ['base_prompt_template', 'explain_base_prompt_template']
metrics_data = {}
for prompt in prompts:
    ground_truths = np.loadtxt('results/'+prompt+'/truth_labs.txt')
    preds = np.loadtxt('results/'+prompt+'/preds.txt')
    get_metric_dict('Llama-2_'+prompt, ground_truths, preds)
   
metrics_df = pd.DataFrame.from_dict(metrics_data, orient='index')
print(metrics_df)
