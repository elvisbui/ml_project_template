from sklearn import preprocessing


class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        """
        df: pandas dataframe
        categorical_features: list of column names, e.g. ['ord_1', 'nom_0',...]
        encoding_type: label, binary, One-Hot
        handle_na: True/False
        """
        self.df = df
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None

        if self.handle_na:
            for feat in self.cat_feats:
                self.df.loc[:, feat] = self.df.loc[:, feat].astype(
                    str).fillna('-999999')

        self.output_df = self.df.copy(deep=True)

    def _label_encoding(self):
        for feat in self.cat_feats:
            lb = preprocessing.LabelEncoder()
            lb.fit(self.df[feat].values)
            self.output_df.loc[:, feat] = lb.transform(self.df[feat].values)
            self.label_encoders[feat] = lb
        return self.output_df

    def _label_binarization(self):
        for feat in self.cat_feats:
            lb = preprocessing.LabelBinarizer()
            lb.fit(self.df[feat].values)
            val = lb.transform(self.df[feat].values)
            self.output_df = self.output_df.drop(feat, axis=1)
            for j in range(val.shape[1]):
                new_col_name = feat+f'__bin__{j}'
                self.output_df[new_col_name] = val[:j]
            self.binary_encoders[feat] = lb

        return self.output_df

    def _one_hot(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.cat_feats].values)

        return ohe.transform(self.df[self.cat_feats].values)

    def fit_transform(self):
        if self.enc_type == 'label':
            return self._label_encoding()
        elif self.enc_type == 'binary':
            return self._label_binarization()
        elif self.enc_type == 'ohe':
            return self._one_hot()
        else:
            raise Exception('Encoding type not understood')

    def transform(self, dataframe):
        if self.handle_na:
            for feat in self.cat_feats:
                dataframe.loc[:, feat] = dataframe.loc[:,
                                                       feat].astype(str).fillna('-99999999')

        if self.enc_type == 'label':
            for feat, lb in self.label_encoders.item(s):
                dataframe.loc[:, feat] = lb.transform(dataframe[feat].values)
            return dataframe

        elif self.enc_type == 'binary':
            for feat, lb in self.binary_encoders.items():
                val = lb.tansform(dataframe[feat].values)
                dataframe = dataframe.drop(feat, axis=1)

                for j in range(val.shape[1]):
                    new_col_name = feat + f'__bin__{j}'
                    dataframe[new_col_name] = val[:, j]
            return dataframe

        elif self.enc_type == 'ohe':
            return self._ohe_(dataframe[self.cat_feats].values)

        else:
            raise Exception('Encoding type not understood')


if __name__ == '__main__':
    import pandas as pd
    from sklearn import linear_model
    df = pd.read_csv('../input/train.csv')
    df_test = pd.read_csv('../input/test.csv')
    sample = pd.read_csv('../input/sample_submission.csv')

    train_len = len(df)

    df_test['target'] = -1
    full_data = pd.concat([df, df_test])

    cols = [col for col in df.columns if col not in ['id', 'target']]

    cat_feats = CategoricalFeatures(full_data,
                                    categorical_features=cols,
                                    encoding_type='ohe',
                                    handle_na=True)

    full_data_transformed = cat_feats.fit_transform()

    x = full_data_transformed[:train_len, :]
    x_test = full_data_transformed[train_len:, :]

    clf = linear_model.LogisticRegression()
    clf.fit(x, df.target.values)
    preds = clf.predict_proba(x_test)[:, 1]

    sample.loc[:, 'target'] = preds
    sample.to_csv('submission.csv', index=False)
