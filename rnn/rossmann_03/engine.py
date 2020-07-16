import sys
import math
import torch
import datetime
from torch.utils.data import DataLoader
from typing import List
from .model import SalesRNN
from .dataset import FINAL_DAY_TEST, FINAL_DAY_VAL
from ..logger.logger import MetricLogger, SmoothedValue
from rnn.utils import plot_predictions
import numpy as np
import pandas as pd


def train_one_epoch(
        model: SalesRNN, optimizer, criterion, data_loader, device, epoch,
        print_freq=100
):
    model.train()

    metric_logger = MetricLogger(
        delimiter="  ",
        meters_printable=None,
        smoothed_value_window_size=print_freq,
        smoothed_value_fmt="{median:.4f} ({global_avg:.4f})")

    for idx, param_gr in enumerate(optimizer.param_groups):
        metric_logger.add_meter(
            f'lr_{idx}', SmoothedValue(window_size=1, fmt='{value:.6f}')
        )
    header = 'Epoch: [{}]'.format(epoch)

    last_hidden_states = []

    for train, _, _, hidden in metric_logger.log_every(data_loader, print_freq, header):

        for k, v in train.items():
            train[k] = v.to(device)
        hidden = hidden.to(device)

        output, last_hidden = model(train, hidden)

        last_hidden_states.append(last_hidden.detach().clone())

        loss = criterion(output * train['mask'], train['y'] * train['mask'])

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(**dict((f'lr_{idx}', el['lr'])
                                    for idx, el in enumerate(optimizer.param_groups))
                             )

    last_hidden_states = torch.cat(last_hidden_states, dim=1)

    return metric_logger, last_hidden_states


@torch.no_grad()
def evaluate_one_epoch(
        model: SalesRNN, criterion, data_loader, device, epoch, print_freq=10
):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Eval Epoch: [{}]'.format(epoch)

    for _, val, _, hidden in metric_logger.log_every(data_loader, print_freq, header):
        for k, v in val.items():
            val[k] = v.to(device)
        hidden = hidden.to(device)

        output, _ = model(val, hidden)

        loss = criterion(output * val['mask'], val['y'] * val['mask'])

        metric_logger.update(loss=loss.item())

    return metric_logger


@torch.no_grad()
def train_val_ts_prediction(
        model, ds, collate_fn, batch_size, t_train, device,
        store_ids: List[int] = None
):
    """ Prediction for the train-val dataset;
    - one step ahead
    - feed the model with the correct past sales

    :param model:
    :param ds:
    :param collate_fn:
    :param batch_size:
    :param t_train:
    :param device:
    :param store_ids: list of store ids for which the model predictions
        will be plotted
    :return:
    """
    with torch.no_grad():
        model.eval()

        ds.reset_hidden()
        dl = DataLoader(ds, batch_size, shuffle=False, collate_fn=collate_fn)

        last_hidden_states = []
        outputs_train = []
        outputs_val = []

        for train, _, _, hidden in dl:

            for k, v in train.items():
                train[k] = v.to(device)
            hidden = hidden.to(device)

            output, last_hidden = model(train, hidden)

            last_hidden_states.append(last_hidden.detach().clone())
            outputs_train.append(output.detach().clone())

        last_hidden_states = torch.cat(last_hidden_states, dim=1)
        outputs_train = torch.cat(outputs_train, dim=1)

        ds.set_hidden(last_hidden_states)
        dl = DataLoader(ds, batch_size, shuffle=False, collate_fn=collate_fn)

        for _, val, _, hidden in dl:
            for k, v in val.items():
                val[k] = v.to(device)
            hidden = hidden.to(device)

            output, _ = model(val, hidden)

            outputs_val.append(output.detach().clone())

        outputs_val = torch.cat(outputs_val, dim=1)
        outputs = torch.cat((outputs_train, outputs_val), dim=0)
        ys_hat = outputs.permute(1, 0, 2).squeeze(-1).numpy()  # -> (N, L_tr_val)

        ys = np.concatenate((
            ds.train['y'].permute(1, 0, 2).squeeze(-1).numpy(),
            ds.val['y'].permute(1, 0, 2).squeeze(-1).numpy()
        ), axis=-1)  # -> (N, L_tr_val)

        ms = np.concatenate((
            ds.train['mask'].permute(1, 0, 2).squeeze(-1).bool().numpy(),
            ds.val['mask'].permute(1, 0, 2).squeeze(-1).bool().numpy()
        ), axis=-1)  # -> (N, L_tr_val)

        time = np.arange(1, ys.shape[-1] + 1)

        figs = [
            plot_predictions(time[m], y[m], y_hat[m], t_train, idx)
            for idx, (m, y_hat, y) in enumerate(zip(ms, ys_hat, ys))
            if (store_ids is None or idx in store_ids)
        ]

        return figs


@torch.no_grad()
def complete_ts_prediction(
        model, ds, collate_fn, batch_size, device,
):
    """ Prediction for the train-val-test dataset;
    - one step ahead
    - feed the model with the correct past sales (for train-val)
      and with the predicted sales (for test)

    """
    with torch.no_grad():
        model.eval()

        ds.reset_hidden()
        dl = DataLoader(ds, batch_size, shuffle=False, collate_fn=collate_fn)
        last_hidden_states = []

        for train, _, _, hidden in dl:

            for k, v in train.items():
                train[k] = v.to(device)
            hidden = hidden.to(device)

            _, last_hidden = model(train, hidden)

            last_hidden_states.append(last_hidden.detach().clone())

        last_hidden_states = torch.cat(last_hidden_states, dim=1).to(device)

        ds.set_hidden(last_hidden_states.detach().clone())
        dl = DataLoader(ds, batch_size, shuffle=False, collate_fn=collate_fn)
        last_hidden_states = []

        for _, val, _, hidden in dl:
            for k, v in val.items():
                val[k] = v.to(device)
            hidden = hidden.to(device)

            _, last_hidden = model(val, hidden)

            last_hidden_states.append(last_hidden.detach().clone())

        last_hidden_states = torch.cat(last_hidden_states, dim=1).to(device)

        ds.set_hidden(last_hidden_states.detach().clone())
        dl = DataLoader(ds, batch_size, shuffle=False, collate_fn=collate_fn)

        outputs_test = []

        for _, _, test, hidden in dl:
            for k, v in test.items():
                test[k] = v.to(device)
            hidden = hidden.to(device)
            output = None

            batch_outputs_test = []

            for t in range(test['y'].size(0)):
                test_t = dict((k, v[[t], ...]) for k, v in test.items())
                if output is not None:
                    test_t['feature_y_m1'] = output

                output, hidden = model(test_t, hidden)  # (1, N_b, 1), (S, N_b, rnn_hidden)

                batch_outputs_test.append(output)

            batch_outputs_test = torch.cat(batch_outputs_test, dim=0)  # (L, N_b, 1)
            outputs_test.append(batch_outputs_test)

        outputs_test = torch.cat(outputs_test, dim=1).to(torch.device('cpu'))  # (L, N, 1)

        # get a pandas dataframe with the predictions
        #
        idx = pd.date_range(FINAL_DAY_VAL + datetime.timedelta(days=1),
                            FINAL_DAY_TEST, freq='D', name='Date')
        n_stores = outputs_test.size(1)

        df = pd.DataFrame([], index=idx)
        for i in range(n_stores):
            df[i + 1] = outputs_test[:, i, :].squeeze(-1).numpy()

        df = df.reset_index()
        df = pd.melt(
            df,
            id_vars=['Date'],
            value_vars=list(filter(lambda x: x is not 'Date', df.columns)),
            var_name='Store',
            value_name='Sales'
        )
        df['Date'] = df['Date'].apply(str)
        df['Store'] = df['Store'].apply(str)

        df_test = ds.df_tt[ds.df_tt['Date'] > FINAL_DAY_VAL].copy()
        df_test['Date'] = df_test['Date'].apply(str)
        df_test['Store'] = df_test['Store'].apply(str)
        df_test = df_test.drop(['Sales'], 1)

        prediction = df_test \
            .join(df.set_index(['Date', 'Store']), on=['Date', 'Store'], how='left')

        prediction['Sales'] = prediction['Sales'] * prediction['Sales_max']

        prediction = prediction[['Id', 'Sales']]

        return prediction
