import torch
import numpy as np
import pandas as pd
import utils

DEVICE = 'cuda'
EPOCHS = 100


def run_training(fold):
    df = pd.read_csv('train_features.csv')
    df = utils.process_data(df)
    folds = pd.read_csv('train_folds.csv')

    targets = folds.drop(['sig_id', 'kfolds'], axis=1).columns
    features = df.drop('sig_id', axis=1).columns

    df = df.merge(folds, on='sig_id', how='left')

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    x_train = train_df[features].to_array()
    x_valid = valid_df[features].to_array()

    y_train = train_df[targets].to_array()
    y_valid = valid_df[targets].to_array()

    train_dataset = utils.Moadataset(x_train, y_train)
    train_loader = torch.utils.data.Dataloader(
        train_dataset, batch_size=1024, num_workers=8
    )

    model = utils.Mode(...)
    model.to(DEVICE)

    optimizer = torch.optia.Adam(model.parameters(), lr=3e-4)
    # schedular = each batch or each epoch?
    eng = utils.Engine(
        model, optimizer, DEVICE
    )

    for _ in range(EPOCHS):
        train_loss = eng.train(train_loss)
        # valid_loss =
