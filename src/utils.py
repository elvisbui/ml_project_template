import torch
import torch.nn as nn
import pandas as pd


class MoaDataset:
    def __init__(self, dataset, features):
        self.dataset = dataset
        self.features = features

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item):
        return {
            'x': torch.tensor(self.dataset[item, :], dtype=torch.float),
            'y': torch.tensor(self.features[item, :], dtype=torch.float)
        }


class Engine:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    @staticmethod
    def loss_fn(targets, outputs):
        return nn.BCEWithLogitsLoss()(outputs, targets)

    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for data in data_loader:
            self.optimizer.zero.grad()
            inputs = data['x'].to(self.device)
            targets = data['y'].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(targets, outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss/len(data_loader)

    def validate(self, data_loader):
        self.model.eval()
        final_loss = 0
        for data in data_loader:
            inputs = data['x'].to(self.device)
            targets = data['y'].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(targets, outputs)
            final_loss += loss.item()
        return final_loss/len(data_loader)


def add_dummies(data, column):
    ohe = pd.get_dummies(data[column])
    ohe_columns = [f'{column}_{c}' for c in ohe.columns]
    ohe.columns = ohe_columns
    data = data.drop(column, axis=1)
    data = data.join(ohe)
    return data


def process_data(df):
    df = add_dummies(df, 'cp_time')
    df = add_dummies(df, 'cp_dose')
    df = add_dummies(df, 'cp_type')


class Model(nn.Nodule):

    def __init__(self, num_features, num_targets):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_targets)
        )

    def forward(self, x):
        x = self.model(x)
        return x
