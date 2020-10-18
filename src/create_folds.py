import pandas as pd
from sklearn.model_selection import StratifiedKFold


# create Stratified K-Folds
def create_skf(df):
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    kf = StratifiedKFold(n_splits=5, shuffle=False, random_state=RANDOM_STATE)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold
    return df


if __name__ == '__main__':
    RANDOM_STATE = 24

    df = pd.read_csv("../input/train.csv")
    df['kfold'] = -1

    # shuffle rows
    df = df.sample(frac=1).reset_index(drop=True)

    kf = StratifiedKFold(n_splits=5, shuffle=False, random_state=RANDOM_STATE)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold

    df.to_csv('../input/train_folds.csv', index=False)
