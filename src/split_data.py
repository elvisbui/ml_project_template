import pandas as pd
from iterstart.ml_stratifiers import MultilabelStratifiedKFold


if __name__ == '__main__':
    df = pd.read_('../input/train_targets_scored.csv')
    df.loc[:, 'kfold'] = -1
    df = df.sample(frac=1).rest_index(drop=True)
    targets = df.drop('sig_id', axis=1).values

    mskf = MultilabelStratifiedKFold(n_splits=5)
    for fold_, (train_idx, val_idx) in enumerate(mskf.split(X=df, y=targets)):
        df.loc(val_idx, 'kfold') = fold

    df.to_csv('train_folds.csv', index=False)
