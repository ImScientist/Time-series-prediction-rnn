import torch.nn as nn


class TS1RNN(nn.Module):
    def __init__(
            self, rnn_input_size, rnn_hidden_size, rnn_num_layers: int = 2,
            add_linear: bool = False
    ):
        """
        :param rnn_input_size: number of features for every time step
        :param rnn_hidden_size: size of the hidden vector
        :param rnn_num_layers: number of vertically stacked RNN layers
        :param add_linear: add a linear term to the model
        """
        super(TS1RNN, self).__init__()
        self.rnn_input_size = rnn_input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers

        self.rnn = nn.GRU(rnn_input_size, rnn_hidden_size, rnn_num_layers)

        self.conv = nn.Conv1d(
            in_channels=rnn_hidden_size, out_channels=1, kernel_size=1)

        if add_linear is True:
            self.linear = nn.Linear(in_features=rnn_input_size, out_features=1)
        else:
            self.linear = None

    def forward(self, data_in, hidden):
        # data_in -> (L, N, rnn_input_size)
        # hidden -> (S=rnn_num_layers, N, rnn_hidden_size)

        # -> (L, N, rnn_hidden_size), (S, N, rnn_hidden_size)
        out, last_hidden = self.rnn(data_in, hidden)

        # -> (N, rnn_hidden_size, L)
        out = out.permute(1, 2, 0)

        # -> (N, 1, L)
        out = self.conv(out)

        # -> (L, N, 1)
        out = out.permute(2, 0, 1)

        if self.linear is not None:
            out = out + self.linear(data_in)

        # -> (L, N, 1), (S, N, rnn_hidden_size)
        return out, last_hidden
