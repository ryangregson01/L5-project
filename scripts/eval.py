import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import sys

def get_metric_dict(method, labels, preds):
    acc = accuracy_score(labels, preds)
    bac = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    metric_dict = {'accuracy': acc, 'f1_score': f1, 'balanced accuracy': bac}
    metrics_data[method] = metric_dict


def evaluation_summary(description, true_labels, predictions):
    target_classes = ['Non-sensitive (0)', 'Sensitive (1)']
    #print(classification_report(true_labels, predictions, digits=3, zero_division=0, target_names=target_classes))
    confusionMatrix = confusion_matrix(true_labels, predictions)
    fig = plt.figure(1, figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=target_classes)
    disp.plot()
    disp.figure_.savefig("confusion_matrix.pdf")
    #plt.show()


folder_name = sys.argv[1]
prompts = ['b1', 'b2', 'b3'] #['base', 'persona', 'cot']
metrics_data = {}
for prompt in prompts:
    ground_truths = np.loadtxt(f'results/{folder_name}/{prompt}/truth_labs.txt')
    preds = np.loadtxt(f'results/{folder_name}/{prompt}/preds.txt')
    get_metric_dict(folder_name+'_'+prompt, ground_truths, preds)
    #evaluation_summary(f'results/{folder_name}/{prompt}/', ground_truths, preds)
   
metrics_df = pd.DataFrame.from_dict(metrics_data, orient='index')
print('Main metrics:')
print(metrics_df)
metrics_df.to_csv(f'results/{folder_name}.csv', index=True)
