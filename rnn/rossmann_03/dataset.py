import os
import datetime
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import pandas as pd

FIRST_DAY_TRAIN = datetime.datetime(2013, 1, 1)
FINAL_DAY_VAL = datetime.datetime(2015, 7, 31)
FINAL_DAY_TEST = datetime.datetime(2015, 9, 17)


def collate_fn(batch):
    b_train = defaultdict(list)
    b_val = defaultdict(list)
    b_test = defaultdict(list)
    b_hidden = []

    for train, val, test, hidden in batch:
        b_hidden.append(hidden)
        for k, v in train.items():
            b_train[k].append(v)
        for k, v in val.items():
            b_val[k].append(v)
        for k, v in test.items():
            b_test[k].append(v)

    b_train = dict(b_train)
    b_val = dict(b_val)
    b_test = dict(b_test)

    b_hidden = torch.cat(b_hidden, dim=1)  # b_hidden -> (L, N, ?)
    for k, v in b_train.items():
        b_train[k] = torch.cat(v, dim=1)  # b_train[k] -> (L, N, ?)
    for k, v in b_val.items():
        b_val[k] = torch.cat(v, dim=1)  # b_val[k] -> (L, N, ?)
    for k, v in b_test.items():
        b_test[k] = torch.cat(v, dim=1)  # b_test[k] -> (L, N, ?)

    return b_train, b_val, b_test, b_hidden


def add_missing_days(df, fillna_value=0):
    idx = pd.date_range(FIRST_DAY_TRAIN, FINAL_DAY_TEST, freq='D')

    df = df \
        .set_index(['Date']) \
        .resample('D')  \
        .mean() \
        .reindex(idx)

    df['Store'] = df['Store'] \
        .fillna(df['Store'].min())

    df = df.fillna(fillna_value)

    return df


def get_lagged_features(df, col='Open', ds=(1, 2, 3,), fillna_value=0):
    df = df[['Date', 'Store', col]] \
        .groupby(['Store'], group_keys=False) \
        .apply(lambda x: add_missing_days(x, fillna_value))

    for d in ds:
        df[f'{col}_{d}_ahead'] = df \
            .groupby(['Store'], group_keys=False)[col] \
            .shift(-d) \
            .fillna(fillna_value)

    cols = [f'{col}_{d}_ahead' for d in ds]

    df['list_cols'] = df[cols].apply(list, 1)

    df = df.reset_index().rename(columns={'index': 'Date'})

    df = pd.pivot_table(
        df,
        values='list_cols',
        index=['Date'],
        columns=['Store'],
        aggfunc=lambda x: list(x)[0]
    ).sort_index()

    # -> (L, N, # lagged features)
    return torch.tensor(df.values.tolist(), dtype=torch.float32)


def get_time_features():
    days = pd.date_range(FIRST_DAY_TRAIN, FINAL_DAY_TEST, freq='D')
    weekdays = days.map(lambda x: x.weekday())
    weekdays = torch.tensor(weekdays, dtype=torch.float32).view(-1, 1, 1)
    # -> (L, 1, 1)

    return weekdays


def get_store_ohe_features(df):
    columns = [
        'Assortment_a', 'Assortment_b', 'Assortment_c',
        'StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d'
    ]

    return torch.tensor(df[columns].values, dtype=torch.float32).unsqueeze(0)


def get_data(data_dir, n_stores=10e5):
    dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')
    state_holiday_map = {'0': 0, 'a': 1, 'b': 1, 'c': 1}

    df_store = pd.read_csv(os.path.join(data_dir, 'store.csv'))
    df_store = df_store.join(
        pd.concat([
            pd.get_dummies(df_store['Assortment'], prefix='Assortment'),
            pd.get_dummies(df_store['StoreType'], prefix='StoreType')
        ], axis=1)
    )
    df_store = df_store[df_store['Store'] <= n_stores].sort_values(by=['Store'])

    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')

    df_train = pd.read_csv(train_path, parse_dates=[2], date_parser=dateparse, low_memory=False)
    df_test = pd.read_csv(test_path, parse_dates=[3], date_parser=dateparse)# .drop(['Id'], 1)

    df_train = df_train.loc[df_train['Store'] <= n_stores]
    df_test = df_test.loc[df_test['Store'] <= n_stores]

    df_train['Id'] = -1
    df_test['Sales'] = -1
    df_test['Customers'] = 0

    df_tt = pd.concat([df_train, df_test], axis=0) \
        .sort_values(by=['Store', 'Date']) \
        .reset_index(drop=True)

    df_tt['StateHoliday'] = df_tt['StateHoliday'].map(state_holiday_map)

    # Add some features
    #
    sales_max = df_tt \
        .groupby(by=['Store'], group_keys=False) \
        .apply(lambda x: x['Sales'].max()) \
        .rename('Sales_max')

    df_tt = df_tt.join(sales_max, on='Store')
    df_tt['Sales'] = df_tt['Sales'] / df_tt['Sales_max']

    return df_store, df_tt


class RossmannDataset(Dataset):

    def __init__(
            self,
            data_dir: str,
            n_stores: int = 10e5,
            t_train: int = 700,
            t_val: int = len(pd.date_range(FIRST_DAY_TRAIN, FINAL_DAY_VAL, freq='D')),
            t_test: int = len(pd.date_range(FIRST_DAY_TRAIN, FINAL_DAY_TEST, freq='D')),
            t_min_accum: int = 100,
            rnn_hidden_size: int = 32,
            rnn_num_layers: int = 2,
    ):
        """
        :param data_dir:
        :param n_stores: take only the data of the first n_store
        :param t_train: time index vor train-validation splitting
        :param t_min_accum: min amount of non-zero sales days in the past in order to
            include a sales prediction for the next day in the loss function
        :param rnn_hidden_size: size of the hidden vector in the rnn
        :param rnn_num_layers: number of rnn layers stacked on top of each other
        """
        self.t_min_accum = t_min_accum
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers

        df_store, df_tt = get_data(data_dir, n_stores)
        self.df_store = df_store
        self.df_tt = df_tt

        self.data = dict()

        N = len(self.df_tt['Store'].unique())
        L = len(pd.date_range(FIRST_DAY_TRAIN, FINAL_DAY_TEST, freq='D'))
        self.N = N
        self.L = L

        # -> (L, N, 1)
        print('idx')
        self.data['idx'] = torch.tensor(df_store['Store'].values).view(1, -1, 1) \
            .repeat(L, 1, 1)

        # -> (1, N, 7) -> (L, N, 7)
        print('feature_store_ohe')
        self.data['feature_store_ohe'] = get_store_ohe_features(self.df_store) \
            .repeat(L, 1, 1)

        # -> (L, 1, 1) -> (L, N, 1)
        print('feature_time')
        self.data['feature_time'] = get_time_features() \
            .repeat(1, N, 1)

        # -> (L, N, 4)
        print('feature_open')
        self.data['feature_open'] = get_lagged_features(
            self.df_tt, col='Open', ds=(0, 1, 2, 3, 4, 5,), fillna_value=0)

        # -> (L, N, 4)
        print('feature_state_holiday')
        self.data['feature_state_holiday'] = get_lagged_features(
            self.df_tt, col='StateHoliday', ds=(0, 1, 2, 3,), fillna_value=0)

        # -> (L, N, 4)
        print('feature_school_holiday')
        self.data['feature_school_holiday'] = get_lagged_features(
            self.df_tt, col='SchoolHoliday', ds=(0, 1, 2, 3,), fillna_value=0)

        # -> (L, N, 1)
        print('feature_promo')
        self.data['feature_promo'] = get_lagged_features(
            self.df_tt, col='Promo', ds=(0,), fillna_value=0)

        # -> (L, N, 1)
        print('feature_y_m1')
        self.data['feature_y_m1'] = get_lagged_features(
            self.df_tt, col='Sales', ds=(-1,), fillna_value=0)

        # -> (L, N, 1)
        print('y')
        self.data['y'] = get_lagged_features(
            self.df_tt, col='Sales', ds=(0,), fillna_value=0)

        # -> (L, N, 1)
        print('open')
        self.data['open'] = get_lagged_features(
            self.df_tt, col='Open', ds=(0,), fillna_value=0)

        self.train = dict()
        self.val = dict()
        self.test = dict()

        for k, v in self.data.items():
            self.train[k] = v[:t_train, ...]
            self.val[k] = v[t_train:t_val, ...]
            self.test[k] = v[t_val:, ...]

        self.train['mask'] = (self.train['open'] > 0).float() * \
                             ((self.train['open'] > 0).cumsum(dim=0) > self.t_min_accum)
        self.val['mask'] = (self.val['open'] > 0).float()
        self.test['mask'] = (self.test['open'] > 0).float()

        self.hidden = torch.zeros(
            self.rnn_num_layers, self.N, self.rnn_hidden_size, dtype=torch.float32)

    def __getitem__(self, idx):
        """ idx: store_id
        """
        data_train = dict((k, v[:, [idx], :]) for k, v in self.train.items())
        data_val = dict((k, v[:, [idx], :]) for k, v in self.val.items())
        data_test = dict((k, v[:, [idx], :]) for k, v in self.test.items())
        hidden = self.hidden[:, [idx], :]

        return data_train, data_val, data_test, hidden

    def __len__(self):
        return self.N

    def reset_hidden(self):
        self.hidden = torch.zeros(self.rnn_num_layers, self.N, self.rnn_hidden_size)

    def set_hidden(self, hidden):
        assert torch.Size(
            [self.rnn_num_layers, self.N, self.rnn_hidden_size]) == hidden.size()

        self.hidden = hidden
