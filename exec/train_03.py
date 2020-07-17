import os
import torch
import torch.nn as nn

import argparse
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from rnn.rossmann_03.dataset import RossmannDataset, collate_fn
from rnn.rossmann_03.model import SalesRNN
from rnn.rossmann_03.engine import \
    train_one_epoch, evaluate_one_epoch, \
    train_val_ts_prediction, complete_ts_prediction


def train_model(
        data_dir: str,
        model_dir: str,
        log_dir: str,
        store_id_embedding_dim: int = 3,
        rnn_hidden_size: int = 32,
        rnn_num_layers: int = 2,
        batch_size: int = 10,
        num_epochs: int = 10,
        n_stores: int = 10e5,
        t_train: int = 700,
        t_min_accum: int = 100
):
    torch.manual_seed(11)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # save figures with the real and predicted sales for these stores
    store_ids = [4, 8]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ds = RossmannDataset(
        data_dir=data_dir,
        n_stores=n_stores,
        t_train=t_train,
        t_min_accum=t_min_accum,
        rnn_hidden_size=rnn_hidden_size,
        rnn_num_layers=rnn_num_layers
    )

    model = SalesRNN(
        store_id_embedding_dim=store_id_embedding_dim,
        feature_store_ohe_dim=7,  # fixed
        feature_time_dim=1,  # fixed
        feature_open_dim_in=6,
        feature_open_dim_out=2,
        feature_state_holiday_dim=4,
        feature_school_holiday_dim=4,
        feature_holiday_dim_out=2,
        rnn_hidden_size=rnn_hidden_size,
        rnn_num_layers=rnn_num_layers,
        n_stores=n_stores+1  # the store_id starts from 1...
    )
    model.to(device)

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': 1e-2, 'weight_decay': 1e-5},
    ])

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.2, patience=40, verbose=True
    )

    criterion = nn.L1Loss()

    for epoch in range(num_epochs):

        ds.reset_hidden()
        dl = DataLoader(ds, batch_size, shuffle=False, collate_fn=collate_fn)

        logger_train, last_hidden_states = train_one_epoch(
            model, optimizer, criterion, dl, device, epoch, print_freq=20)

        ds.set_hidden(last_hidden_states)
        dl = DataLoader(ds, batch_size, shuffle=False, collate_fn=collate_fn)

        logger_val = evaluate_one_epoch(
            model, criterion, dl, device, epoch, print_freq=20)

        lr_scheduler.step(metrics=logger_val.meters['loss'].value)

        if epoch % 40 == 0:
            figs = train_val_ts_prediction(
                model, ds, collate_fn, batch_size, t_train, device, store_ids
            )

            df = complete_ts_prediction(
                model, ds, collate_fn, batch_size, device,
            )

            df.to_csv(os.path.join(model_dir, f'predictions_epoch_{epoch}.csv'),
                      index=False)

            torch.save(
                model.state_dict(),
                os.path.join(model_dir, f'model_state_dict_epoch_{epoch}.pth')
            )
        else:
            figs = None

        with SummaryWriter(log_dir) as w:
            result_dict = defaultdict(dict)

            for k, meter in logger_train.meters.items():
                result_dict[k].update({'train': meter.global_avg})

            for k, meter in logger_val.meters.items():
                result_dict[k].update({'val': meter.global_avg})

            for k, v in result_dict.items():
                w.add_scalars(k, v, epoch)

            if figs is not None:
                for idx, fig in enumerate(figs):
                    w.add_figure(f'predictions_{idx}', fig, epoch)

            for name, params in model.named_parameters():
                w.add_histogram(name, params.data, epoch)

    print('FINE')


if __name__ == "__main__":
    """ 
    Example:         
        python exec/train_03.py \
            --data_dir ${DATA_DIR} \
            --model_dir ${RESULTS_DIR}/models_rossmann_02 \
            --log_dir ${RESULTS_DIR}/logs_rossmann_02 \
            --store_id_embedding_dim 4 \
            --num_epochs 201 \
            --rnn_hidden_size 32 \
            --rnn_num_layers 4 \
            --batch_size 4 \
            --n_stores 12 \
            --t_train 708
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, dest='data_dir', help='data directory')
    parser.add_argument('--model_dir', type=str, dest='model_dir',
                        help='Directory where the model that will be loaded is saved.')
    parser.add_argument('--log_dir', type=str, dest='log_dir',
                        help='Directory where the training logs will be saved.')
    parser.add_argument('--store_id_embedding_dim', type=int, dest='store_id_embedding_dim', default=4,
                        help='Embedding dimension of the store id')
    parser.add_argument('--rnn_hidden_size', type=int, dest='rnn_hidden_size', default=32,
                        help='Dimension of the hidden state of the rnn')
    parser.add_argument('--rnn_num_layers', type=int, dest='rnn_num_layers', default=2,
                        help='Number of rnn layers stacked on top of each other.')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=10,
                        help='batch_size (number of stores per batch)')
    parser.add_argument('--num_epochs', type=int, dest='num_epochs', default=10,
                        help='Numbers of training epochs.')
    parser.add_argument('--n_stores', type=int, dest='n_stores', default=None,
                        help='Numbers of stores used in the model.')
    parser.add_argument('--t_train', type=int, dest='t_train', default=100,
                        help='Training size; Use the first t_train days of the time '
                             'series for model training and the rest for validation.')

    args = parser.parse_args()

    train_model(**vars(args))
