from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from dataset import load_sara
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, fbeta_score, roc_auc_score
import pandas as pd
import re
import email
import gensim


def ml_preproc(s, tokenizer, c_size=2048):
    def preprocess(e):
        message = email.message_from_string(e)
        clean = message.get_payload()
        clean = re.sub('\S*@\S*\s?', '', clean)
        clean = re.sub('\s+', ' ', clean)
        clean = re.sub("\'", "", clean)
        clean = gensim.utils.simple_preprocess(str(clean), deacc=True, min_len=1, max_len=100) 
        return clean

    def remove_doubles(df):
        already_exists = []
        unique_df = []
        for i, s in enumerate(df.iterrows()):
            idd = s[1].doc_id
            text = s[1].text
            sensitivity = s[1].sensitivity
            if text in already_exists:
                continue
            already_exists.append(text)
            unique_df.append({'doc_id': idd, 'text':text, 'sensitivity':sensitivity})    
        return pd.DataFrame.from_dict(unique_df)

    def main(s, tokenizer):
        processed_emails = [preprocess(a) for a in s.text]
        ids = s.doc_id.tolist()
        sens = s.sensitivity.tolist()
        texts = []
        for i, text in enumerate(s.text):
            new_email = ' '.join(processed_emails[i])
            texts.append(new_email)

        new_dict = {'doc_id': ids, 'text': texts, 'sensitivity':sens}
        preproc_df = pd.DataFrame.from_dict(new_dict)
        preproc_df = remove_doubles(preproc_df)
        new_docs = pd.DataFrame.from_dict(preproc_df)
        return new_docs

    return main(s, tokenizer)


def ml(train, train_labels, test, test_labels, id_list):
    df_list = []

    vectorizer = CountVectorizer()
    vectorizer.fit(train)
    train_features = vectorizer.transform(train)
    test_features = vectorizer.transform(test)

    dummy_mf = DummyClassifier(strategy="most_frequent")
    dummy_mf.fit(train_features, train_labels)
    mf_test_preds = dummy_mf.predict(test_features)
    get_metric_dict("Dummy MF", test_labels, mf_test_preds)
    df_list += ml_df("MF", mf_test_preds, test_labels, id_list)


    dummy_rand = DummyClassifier(strategy="stratified")
    dummy_rand.fit(train_features, train_labels)
    rand_test_preds = dummy_rand.predict(test_features)
    get_metric_dict("Dummy Random stratified sampling", test_labels, rand_test_preds)
    df_list += ml_df("Rand", rand_test_preds, test_labels, id_list)


    tfidf_vectoizer = TfidfVectorizer()
    train_tfidf = tfidf_vectoizer.fit_transform(train)
    test_tfidf = tfidf_vectoizer.transform(test)

    lr = LogisticRegression(max_iter=500)
    lr.fit(train_tfidf, train_labels)
    lr_preds = lr.predict(test_tfidf)
    get_metric_dict("LR", test_labels, lr_preds)
    df_list += ml_df("LR", lr_preds, test_labels, id_list)

    svm_model = SVC()
    svm_model.fit(train_tfidf, train_labels)
    svm_preds = svm_model.predict(test_tfidf)
    get_metric_dict("SVM", test_labels, svm_preds)
    df_list += ml_df("SVM", svm_preds, test_labels, id_list)

    df = pd.DataFrame(df_list)
    return df


def get_metric_dict(method, labels, preds):
    acc = accuracy_score(labels, preds)
    bac = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='binary')
    prec = precision_score(labels, preds, average='binary')
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    tpr = tp / (tp+fn)
    tnr = tn / (tn+fp)
    f2 = fbeta_score(labels, preds, average='binary', beta=2)
    metric_dict = {'acc': acc, 'prec':prec, 'TPR':tpr, 'TNR':tnr, 'f1_score':f1, 'f2_score':f2, 'BAC':bac,}
    print(method)
    print(metric_dict)


def ml_df(name, pred, gt, id_list):
    x = []
    for i, v in enumerate(pred):
        x.append({'model': name, 'prediction':v, 'ground_truth':gt[i], 'prompt':name, 'doc_id':id_list[i]})
    return x


def main():
    np.random.seed(1)
    s = load_sara()
    p = ml_preproc(s, '')
    X = p.text.to_numpy()
    y = p.sensitivity.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1)

    id_list = []
    for val in X_test:
        find = p[p.text==val]
        id_list.append(find.iloc[0].doc_id)
    majority_class_index = np.where(y_train == 0)[0]
    minority_class_index = np.where(y_train == 1)[0]
    downsampled_majority_index = np.random.choice(majority_class_index, size=len(minority_class_index), replace=False)
    combined_indices = np.concatenate([downsampled_majority_index, minority_class_index])
    X_train = X_train[combined_indices]
    y_train = y_train[combined_indices]
    v = ml(X_train, y_train, X_test, y_test, id_list)
    return v


x = main()
print(x)