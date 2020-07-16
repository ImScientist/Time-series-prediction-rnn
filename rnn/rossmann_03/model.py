import torch
import torch.nn as nn
from typing import Dict


class SalesRNN(nn.Module):
    def __init__(
            self,
            store_id_embedding_dim: int = 5,
            feature_store_ohe_dim: int = 7,
            feature_time_dim: int = 1,
            feature_open_dim_in: int = 6,
            feature_open_dim_out: int = 2,
            feature_state_holiday_dim: int = 4,
            feature_school_holiday_dim: int = 4,
            feature_holiday_dim_out: int = 2,
            rnn_hidden_size: int = 32,
            rnn_num_layers: int = 2,
            n_stores: int = 1e3,
    ):
        """
        :param data_input_features: number of features of a data point
        :param rnn_hidden_size: size of the hidden vector
        :param rnn_num_layers: number of vertically stacked RNN layers
        :param embedding_dim: embedding dimension of `store id`
        :param n_stores: number of unique stores
        """
        super(SalesRNN, self).__init__()

        self.store_id_embedding_dim = store_id_embedding_dim
        self.feature_store_ohe_dim = feature_store_ohe_dim
        self.feature_time_dim = feature_time_dim
        self.feature_open_dim_in = feature_open_dim_in
        self.feature_open_dim_out = feature_open_dim_out
        self.feature_state_holiday_dim = feature_state_holiday_dim
        self.feature_school_holiday_dim = feature_school_holiday_dim
        self.feature_holiday_dim_out = feature_holiday_dim_out
        self.rnn_input_size = store_id_embedding_dim + \
                              feature_open_dim_out + feature_holiday_dim_out + \
                              feature_store_ohe_dim + feature_time_dim + 1 + 1
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers

        self.store_id_embedding = nn.Embedding(n_stores, store_id_embedding_dim)

        self.feature_open_lin_map = nn.Linear(feature_open_dim_in, feature_open_dim_out)

        self.feature_holiday_lin_map = nn.Linear(feature_state_holiday_dim + feature_school_holiday_dim,
                                                 feature_holiday_dim_out)

        self.rnn = nn.GRU(self.rnn_input_size, self.rnn_hidden_size, self.rnn_num_layers)

        self.conv = nn.Conv1d(in_channels=self.rnn_hidden_size, out_channels=1, kernel_size=1)

    def forward(self, data: Dict, hidden):
        """

        :param data: data[<any key>] -> (L, N, ?)
        :param hidden: hidden -> (S=rnn_num_layers, N, rnn_hidden_size)
        :return:
        """

        # -> (L, N, store_id_embedding_dim)
        f_store_id = self.store_id_embedding(data['idx'].squeeze(-1))

        # -> (L, N, feature_open_dim_out)
        f_open = self.feature_open_lin_map(data['feature_open'])

        # -> (L, N, feature_holiday_dim_out)
        f_holiday = torch.cat((data['feature_state_holiday'], data['feature_school_holiday']), dim=-1)
        f_holiday = self.feature_holiday_lin_map(f_holiday)

        # -> (L, N, feature_store_ohe_dim + feature_time_dim + 1 + 1)
        f_other = torch.cat((
            data['feature_store_ohe'], data['feature_time'],
            data['feature_promo'], data['feature_y_m1']), dim=-1)

        # -> (L, N, store_id_embedding_dim + feature_open_dim_out + feature_holiday_dim_out +
        #           feature_store_ohe_dim + feature_time_dim + 1 + 1)
        f_all = torch.cat((f_store_id, f_open, f_holiday, f_other), dim=-1)

        # -> (L, N, rnn_hidden_size), (S, N, rnn_hidden_size)
        out, last_hidden = self.rnn(f_all, hidden)

        # -> (N, rnn_hidden_size, L)
        out = out.permute(1, 2, 0)

        # -> (N, 1, L)
        out = self.conv(out)

        # -> (L, N, 1)
        out = out.permute(2, 0, 1)

        # Use the common sense that there will be no sells if the store is closed
        out = out * data['open']

        # -> (L, N, 1), (S, N, rnn_hidden_size)
        return out, last_hidden
