import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

from . import dispatcher

TRAINING_DATA = os.environ.get('TRAINING_DATA')
TEST_DATA = os.environ.get('TEST_DATA')
FOLD = int(os.environ.get('FOLD'))
MODEL = os.environ.get('MODEL')

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3],
}

if __name__ == '__main__':
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df[df.kfold == FOLD].reset_index(drop=True)

    y_train = train_df.target.values
    y_valid = valid_df.target.values

    train_df = train_df.drop(['id', 'target', 'kfold'], axis=1)
    valid_df = valid_df.drop(['id', 'target', 'kfold'], axis=1)

    valid_df = valid_df[train_df.columns]

    label_encoders = {}

    for col in train_df.columns:
        lb = preprocessing.LabelEncoder()
        train_df.loc[:, col] = train_df.loc[:, col].astype(str).fillna('NONE')
        valid_df.loc[:, col] = valid_df.loc[:, col].astype(str).fillna('NONE')
        df_test.loc[:, col] = df_test.loc[:, col].astype(str).fillna('NONE')
        lb.fit(train_df[col].values.tolist() +
               valid_df[col].values.tolist() +
               df_test[col].values.tolist())
        train_df.loc[:, col] = lb.transform(train_df[col].values.tolist())
        valid_df.loc[:, col] = lb.transform(valid_df[col].values.tolist())
        label_encoders[col] = lb

    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, y_train)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(metrics.roc_auc_score(y_valid, preds))

    joblib.dump(label_encoders, f'models/{MODEL}_{FOLD}_label_encoder.pkl')
    joblib.dump(clf, f'models/{MODEL}_{FOLD}.pkl')
    joblib.dump(train_df.columns, f'models/{MODEL}_{FOLD}_columns.pkl')
