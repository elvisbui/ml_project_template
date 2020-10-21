import optuna
import torch
from torch import nn
import numpy as np
import pandas as pd

from functools import partial
import utils

DEVICE = 'cuda'
EPOCHS = 100


class ModelX(nn.Module):
    def __init__(self, num_features, num_targets, num_layers, hidden_size, dropout):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            if len(layers) == 0:
                layers.append(nn.Linear(num_features, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                nn.ReLU()
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                nn.ReLU()
        layers.append(nn.Linear(hidden_size, num_targets))

        self.model = nn.Sequentail(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


def run_training(fold, params, save_model=False):
    df = pd.read_csv('../input/train_features.csv')
    df = utils.process_data(df)
    folds = pd.read_csv('../input/train_folds.csv')

    non_scored_df = pd.read_csv('../input/train_targets_nonscored.csv')
    non_scored_targets = non_scored_df.drop(
        'sig_id', axis=1).to_numpy().sum(axis=1)
    non_scored_df.loc[:, 'nscr'] = non_scored_targets
    drop_cols = [
        col for col in non_scored_df.columns if c not in ('nscr', 'sig_id')]
    non_scored_df = non_scored_df.drop(drop_cols, axis=1)
    folds = folds.merge(non_scored_df, on='sig_id', how='left')

    targets = folds.drop(['sig_id', 'kfold'], axis=1).columns
    features = df.drop('sig_id', axis=1).columns

    df = df.merge(folds, on='sig_id', how='left')

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    x_train = train_df[features].to_numpy()
    x_valid = valid_df[features].to_numpy()

    y_train = train_df[targets].to_numpy()
    y_valid = valid_df[targets].to_numpy()

    train_dataset = utils.MoaDataset(x_train, y_train)
    train_loader = torch.utils.data.Dataloader(
        train_dataset, batch_size=1024, num_workers=8
    )

    valid_dataset = utils.MoaDataset(x_valid, y_valid)
    valid_loader = torch.utils.data.Dataloader(
        train_dataset, batch_size=1024, num_workers=8
    )

    model = ModelX(
        num_features=x_train.shape[1],
        num_targets=y_train.shape[1],
        num_layers=params['num_layers'],
        hidden_size=params['hidden_size'],
        dropout=params['dropout']
    )

    model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, threshold=0.00001, mode='min', verbose=True,
    )

    eng = utils.Engine(
        model,
        optimizer,
        device=DEVICE,
    )

    best_loss = np.inf
    early_stopping = 10
    early_stopping_counter = 0

    for epoch in range(EPOCHS):
        train_loss = eng.train(train_loader)
        valid_loss = eng.validate(valid_loader)
        scheduler.step(valid_loss)

        print(
            f'fold={fold}, epoch={epoch}, train_loss={train_loss}, valid_loss={valid_loss}')

        if valid_loss < best_loss:
            best_loss = valid_loss
            if save_model:
                torch.save(model.state_dict(), f'model_fold{fold}.bin')
        else:
            early_stopping_counter += 1

        if early_stopping_counter > early_stopping:
            break

    print(f'fold ={fold}, best validation loss={best_loss}')
    return best_loss


def objective(trial):
    params = {
        'num_layers': trial.suggest_int('num_layers', 1, 7),
        'hidden_size': trial.suggest_int('hidden_size', 16, 2048),
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.8),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    }
    all_loss = []
    for fold in range(5):
        temp_loss = run_training(fold, params, save_model=True)
        all_loss.append(temp_loss)
    return np.mean(all_loss)


if __name__ == '__main__':
    partial_obj = partial(objective)
    study = optuna.create_study(direction='minimize')
    study.optimize(partial_obj, n_trials=150)
    print('Best trial:')
    trial_ = study.best_trial
    print(f'Value: {trial_.value}')
    best_params = trial_.param

    scores = 0
    for j in range(5):
        score = run_training(fold=j, params=best_params, save_model=True)
        scores += score

    print(f'OOF Score: {scores/5}')
