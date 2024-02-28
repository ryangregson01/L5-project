import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, fbeta_score
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def get_metric_dict(labels, preds):
    acc = accuracy_score(labels, preds)
    bac = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    prec = precision_score(labels, preds, average='weighted')
    rec = recall_score(labels, preds, average='weighted')
    f2 = fbeta_score(labels, preds, average='weighted', beta=2)
    metric_dict = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'bal accuracy': bac, 'f2_score': f2}
    return metric_dict


def evaluation_summary(true_labels, predictions):
    target_labels = [0, 1]
    target_classes = ['Non-sensitive (0)', 'Sensitive (1)']
    report = classification_report(true_labels, predictions, labels=target_labels, target_names=target_classes, digits=3, zero_division=0)
    #print(report)
    confusionMatrix = confusion_matrix(true_labels, predictions, labels=target_labels)
    fig = plt.figure(1, figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=target_classes)
    disp.plot()
    #disp.figure_.savefig(description+"confusion_matrix.pdf")
    #plt.show()
    return disp

def jupyter_evaluation(labels, preds):
    metric_data = get_metric_dict(labels, preds)
    metrics_df = pd.DataFrame.from_dict(metric_data, orient='index')
    print('Main metrics:')
    print(metrics_df)
    disp = evaluation_summary(labels, preds)
    plt.show()


def run_evaluation(name):
    folder_name = name
    prompts = ['b1','b2','b1_2','b2_2','b1sys','b2sys','b1_2sys'] #['itspersonal', 'itspersonal_2', 'itspersonalfewshot']#['b1', 'b2', 'b1_2', 'b2_2', 'b1sys', 'b2sys', 'b1_2sys', 'b2_2sys']
    metrics_data = {}
    for prompt in prompts:
        ground_truths = np.loadtxt(f'results/model_results/{folder_name}/{prompt}/truth_labs.txt')
        preds = np.loadtxt(f'results/model_results/{folder_name}/{prompt}/preds.txt')
        metrics_data[folder_name+'_'+prompt] = get_metric_dict(ground_truths, preds)
        cm_path = f'results/model_results/{folder_name}/cm/'
        if not os.path.exists(cm_path):
            os.makedirs(cm_path)
        
        desc = cm_path+prompt
        disp = evaluation_summary(ground_truths, preds)
        disp.figure_.savefig(cm_path+"CM_"+prompt+".pdf")

    metrics_df = pd.DataFrame.from_dict(metrics_data, orient='index')
    metric_path = f'results/metric_overview/'
    if not os.path.exists(metric_path):
        os.makedirs(metric_path)
    metrics_df.to_csv(f'{metric_path}{folder_name}.csv', index=True)

#run_evaluation()
