import os
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib


from . import dispatcher


def predict(test_data_path, model_type, model_path):
    df = pd.read_csv(test_data_path)
    test_idx = df['id'].values
    predictions = None

    for FOLD in range(5):
        df = pd.read_csv(test_data_path)
        encoders = joblib.load(os.path.join(
            model_path, f'{model_type}_{FOLD}_label_encoder.pkl'))
        cols = joblib.load(os.path.join(
            model_path, f'{model_type}_{FOLD}_columns.pkl'))
        for col in encoders:
            lb = encoders[col]
            df.loc[:, col] = df.loc[:, col].astype(str).fillna('NONE')
            df.loc[:, col] = lb.transform(df[col].values.tolist())

        clf = joblib.load(os.path.join(model_path, f'{model_path}_{FOLD}.pkl'))

        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions /= 5

    submision = pd.Dataframe(np.column_stack(
        (test_idx, predictions)), columns=['id', 'target'])
    return submision
