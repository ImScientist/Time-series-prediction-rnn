import sys
import math
import torch
from torch.utils.data import DataLoader
from ..logger.logger import MetricLogger, SmoothedValue
from .utils import plot_predictions


def train_one_epoch(
        model, optimizer, criterion, data_loader, device, epoch,
        print_freq=100, t_warmup=365
):
    model.train()

    metric_logger = MetricLogger(
        delimiter="  ",
        meters_printable=None,
        smoothed_value_window_size=print_freq,
        smoothed_value_fmt="{median:.4f} ({global_avg:.4f})")

    for idx, param_gr in enumerate(optimizer.param_groups):
        metric_logger.add_meter(f'lr_{idx}',
                                SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    last_hidden_states = []

    for x_train, y_train, _, _, hidden in metric_logger.log_every(
            data_loader, print_freq, header):

        x_train = x_train.to(device)
        y_train = y_train.to(device)
        hidden = hidden.to(device)

        output, last_hidden = model(x_train, hidden)

        last_hidden_states.append(last_hidden.detach().clone())

        loss = criterion(output[t_warmup:, ...], y_train[t_warmup:, ...])

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
        model, criterion, data_loader, device, epoch, print_freq=10
):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Eval Epoch: [{}]'.format(epoch)

    for _, _, x_val, y_val, hidden in metric_logger.log_every(
            data_loader, print_freq, header):
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        hidden = hidden.to(device)

        output, _ = model(x_val, hidden)

        loss = criterion(output, y_val)

        metric_logger.update(loss=loss.item())

    return metric_logger


@torch.no_grad()
def complete_ts_prediction(
        model, ds, collate_fn, batch_size, t_warmup, t_train, device
):
    with torch.no_grad():
        model.eval()

        ds.reset_hidden()
        dl = DataLoader(ds, batch_size, shuffle=False, collate_fn=collate_fn)

        last_hidden_states = []
        outputs_train = []
        outputs_val = []

        for x_train, _, _, _, hidden in dl:
            x_train = x_train.to(device)
            hidden = hidden.to(device)

            output, last_hidden = model(x_train, hidden)

            last_hidden_states.append(last_hidden.detach().clone())
            outputs_train.append(output.detach().clone())

        last_hidden_states = torch.cat(last_hidden_states, dim=1)
        outputs_train = torch.cat(outputs_train, dim=1)

        ds.set_hidden(last_hidden_states)
        dl = DataLoader(ds, batch_size, shuffle=False, collate_fn=collate_fn)

        for _, _, x_val, _, hidden in dl:
            x_val = x_val.to(device)
            hidden = hidden.to(device)

            output, _ = model(x_val, hidden)

            outputs_val.append(output.detach().clone())

        outputs_val = torch.cat(outputs_val, dim=1)

        outputs = torch.cat((outputs_train, outputs_val), dim=0)

        y = ds.y_train.mean(1).view(-1).tolist() + \
            ds.y_val.mean(1).view(-1).tolist()

        y_pred = outputs.mean(1).view(-1).tolist()

        time = [idx for idx in range(1, len(y) + 1)]

        fig = plot_predictions(time, y, y_pred, t_warmup, t_train)

        return fig
