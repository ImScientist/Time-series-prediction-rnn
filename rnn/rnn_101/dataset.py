import torch
import numpy as np
from torch.utils.data import Dataset


def collate_fn(batch):
    x_train_batch, y_train_batch, x_val_batch, y_val_batch, hidden_batch = \
        [], [], [], [], []

    for x_train, y_train, x_val, y_val, hidden in batch:
        x_train_batch.append(x_train)
        y_train_batch.append(y_train)
        x_val_batch.append(x_val)
        y_val_batch.append(y_val)
        hidden_batch.append(hidden)

    x_train_batch = torch.cat(x_train_batch, dim=1)
    y_train_batch = torch.cat(y_train_batch, dim=1)
    x_val_batch = torch.cat(x_val_batch, dim=1)
    y_val_batch = torch.cat(y_val_batch, dim=1)
    hidden_batch = torch.cat(hidden_batch, dim=1)

    return x_train_batch, y_train_batch, x_val_batch, y_val_batch, hidden_batch


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(
        season_time < 0.4,
        np.cos(season_time * 2 * np.pi),
        1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def get_timeseries(
        slope, baseline, amplitude, noise_level, n_periods=4, period=365
):
    time = np.arange(n_periods * period + 1)

    series = baseline + \
             trend(time, slope) + \
             seasonality(time, period=period, amplitude=amplitude) + \
             white_noise(time, noise_level, seed=42)

    return series


class TS1DDataset(Dataset):
    """ Single realization of a stochastic process.
    Use a kernel (hidden vectors) with infinite memory => batch size = 1
    """

    def __init__(
            self, slope=0.05, baseline=10, amplitude=40, noise_level=5,
            n_periods=4,
            period=365,
            t_train: int = 3 * 365,
            rnn_num_layers: int = 2,
            rnn_hidden_size: int = 32
    ):
        self.slope = slope
        self.baseline = baseline
        self.amplitude = amplitude
        self.noise_level = noise_level
        self.n_periods = n_periods
        self.period = period

        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers

        # -> (L,)
        data = get_timeseries(
            slope, baseline, amplitude, noise_level, n_periods, period)

        self.min = min(data)
        self.max = max(data)
        data = (data - self.min) / (self.max - self.min)

        # -> (L, 1 = # different measurements of this process, 1 = # features)
        data, data_y = data[:-1], data[1:]
        data, data_y = data.reshape(-1, 1, 1), data_y.reshape(-1, 1, 1)

        self.x_train = torch.tensor(data[:t_train], dtype=torch.float32)
        self.x_val = torch.tensor(data[t_train:], dtype=torch.float32)

        self.y_train = torch.tensor(data_y[:t_train], dtype=torch.float32)
        self.y_val = torch.tensor(data_y[t_train:], dtype=torch.float32)

        self.hidden = torch.zeros(
            self.rnn_num_layers, self.x_train.size(1), self.rnn_hidden_size,
            dtype=torch.float32)

    def __getitem__(self, idx):
        """ idx: measurement number (in the current example we have only a single measurement"""
        x_train = self.x_train[:, [idx], :]
        y_train = self.y_train[:, [idx], :]
        x_val = self.x_val[:, [idx], :]
        y_val = self.y_val[:, [idx], :]
        hidden = self.hidden[:, [idx], :]

        return x_train, y_train, x_val, y_val, hidden

    def __len__(self):
        return self.x_train.size(1)

    def reset_hidden(self):
        self.hidden = torch.zeros(
            self.rnn_num_layers, self.x_train.size(1), self.rnn_hidden_size)

    def set_hidden(self, hidden):
        assert torch.Size([self.rnn_num_layers,
                           self.x_train.size(1),
                           self.rnn_hidden_size]) == hidden.size()

        self.hidden = hidden
