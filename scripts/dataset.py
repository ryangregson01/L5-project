import ir_datasets
import pandas as pd

def get_sara():
    return ir_datasets.load('sara')

def dataset_to_df(dataset):
    doc_ids = []
    doc_text = []
    doc_sens = []
    for doc in dataset.docs_iter():
        doc_ids.append(doc.doc_id)
        doc_text.append(doc.text)
        doc_sens.append(doc.sensitivity)

    sara_dict = {'doc_id':doc_ids, 'text':doc_text, 'sensitivity':doc_sens}
    df = pd.DataFrame.from_dict(sara_dict)
    return df

def load_sara():
    sara_dataset = get_sara()
    sara_df = dataset_to_df(sara_dataset)
    return sara_df

