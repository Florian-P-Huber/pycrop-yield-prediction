from torch import nn
import torch

import math
from pathlib import Path

from .base import ModelBase


class CNN_LSTM_scratchModel(ModelBase):

    def __init__(
        self, in_channels=9, num_bins=32,
        hidden_size=256, rnn_dropout=0.75, dense_features=None, savedir=Path("data/models"), use_gp=True, sigma=1, r_loc=0.5,
        r_year=1.5, sigma_e=0.01, sigma_b=0.01, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

        model = Combine(
        )

        if dense_features is None:
            num_dense_layers = 2
        else:
            num_dense_layers = len(dense_features)
        model_weight = f"dense_layers.{num_dense_layers - 1}.weight"
        model_bias = f"dense_layers.{num_dense_layers - 1}.bias"

        super().__init__(
            model, model_weight, model_bias, "rnn",
            savedir, use_gp, sigma, r_loc, r_year,
            sigma_e, sigma_b, device)

    def reinitialize_model(self, time=None):
        pass
        #self.model.initialize_weights()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.device = torch.device('cuda')
        self.conv_blocks = (nn.Sequential(
            nn.Conv2d(11, 32, (2,1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2,1)),
            nn.Conv2d(32, 64, (2,1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2,1)),
        ).to(self.device))

    def forward(self, x):
        x = self.conv_blocks(x)
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        #return F.log_softmax(x, dim=1)
        return x


class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        self.cnn = CNN()
        self.rnn = nn.LSTM(
            input_size=448, 
            hidden_size=256, 
            num_layers=1,
            batch_first=True)

        self.device = torch.device('cuda')
        self.dense_outs = []
        for i in range(36):
            self.dense_outs.append(nn.Sequential(
                nn.Linear(256,64),
                nn.ReLU()).to(self.device))

        self.dropout = nn.Dropout(0.5)
        self.final = nn.Linear(2304, 1)

        #some crap that is called in the other code -.-
        dense_features = [256, 1]
        dense_features.insert(0, 256)
        self.dense_layers = nn.ModuleList(
            [
                nn.Linear(
                    in_features=dense_features[i - 1], out_features=dense_features[i]
                )
                for i in range(1, len(dense_features))
            ]
        )

    def forward(self, x, return_last_dense=False):
        #[batch, bands, times, bins]
        batch_size, C, timesteps,  H= x.size()
        c_in = x.view(batch_size * timesteps, C, H, 1) #This mimiks keras time distributed
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)


        hidden_state = torch.zeros(1, x.shape[0], 256)
        cell_state = torch.zeros(1, x.shape[0], 256)

        if x.is_cuda:
            hidden_state = hidden_state.cuda()
            cell_state = cell_state.cuda()
        
        hidden_list = []
        for i in range(r_in.shape[1]):
            # The reason the RNN is unrolled here is to apply dropout to each timestep;
            # The rnn_dropout argument only applies it after each layer. This better mirrors
            # the behaviour of the Dropout Wrapper used in the original repository
            # https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/DropoutWrapper
            lstm_in_x = r_in[:, i, :].unsqueeze(1)
            _, (hidden_state, cell_state) = self.rnn(
                lstm_in_x, (hidden_state, cell_state))
            hidden_list.append(self.dense_outs[i](hidden_state))

        out = torch.stack(hidden_list, dim=1).squeeze().permute(1,0,2).contiguous()
        out = out.view(out.shape[0], out.shape[1]*out.shape[2])
        out = self.dropout(out)
        x = self.final(out)

        #r_out, (h_n, h_c) = self.rnn(r_in)
        #r_out2 = self.dense_out(r_out[:, -1, :])
        
        return x


class CNN_LSTM_SCRATCH_NET(nn.Module):
    def __init__(
        self, in_channels=9, num_bins=32, hidden_size=256,
        num_rnn_layers=1, rnn_dropout=0.25, dense_features=None):
        super().__init__()

        if dense_features is None:
            dense_features = [256, 1]
        dense_features.insert(0, hidden_size)

        self.device = torch.device('cuda')
        self.conv_blocks = (nn.Sequential(
            nn.Conv2d(in_channels, 32, (2,1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2,1)),
            nn.Conv2d(32, 64, (2,1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2,1)),
        ).to(self.device))

        self.dropout = nn.Dropout(rnn_dropout)
        self.rnn = nn.LSTM(
            input_size=448,
            hidden_size=hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
        )
        self.hidden_size = hidden_size

        self.dense_layers = nn.ModuleList(
            [
                nn.Linear(
                    in_features=dense_features[i -
                                               1], out_features=dense_features[i]
                )
                for i in range(1, len(dense_features))
            ]
        )

        self.initialize_weights()

    def initialize_weights(self):

        sqrt_k = math.sqrt(1 / self.hidden_size)
        for parameters in self.rnn.all_weights:
            for pam in parameters:
                nn.init.uniform_(pam.data, -sqrt_k, sqrt_k)

        for dense_layer in self.dense_layers:
            nn.init.kaiming_uniform_(dense_layer.weight.data)
            nn.init.constant_(dense_layer.bias.data, 0)

    def forward(self, x, return_last_dense=False):
        """
        If return_last_dense is true, the feature vector generated by the second to last
        dense layer will also be returned. This is then used to train a Gaussian Process model.
        """
        # the model expects feature_engineer to have been run with channels_first=True, which means
        # the input is [batch, bands, times, bins].
        # Reshape to [batch, times, bands * bins]
        #input is [batch, bands, times, bins]
        # we need [batch, times, bands, bins, 1]
        x = x.permute(0, 2, 1, 3)
        x = torch.unsqueeze(x, dim=-1)
        sequence_length = x.shape[1]

        hidden_state = torch.zeros(1, x.shape[0], self.hidden_size)
        cell_state = torch.zeros(1, x.shape[0], self.hidden_size)

        if x.is_cuda:
            hidden_state = hidden_state.cuda()
            cell_state = cell_state.cuda()

        for i in range(sequence_length):
            # The reason the RNN is unrolled here is to apply dropout to each timestep;
            # The rnn_dropout argument only applies it after each layer. This better mirrors
            # the behaviour of the Dropout Wrapper used in the original repository
            # https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/DropoutWrapper
            lstm_in_x = self.conv_blocks(x[:, i, :, :, :])
            lstm_in_x = lstm_in_x.view(lstm_in_x.shape[0], lstm_in_x.shape[1] * lstm_in_x.shape[2], lstm_in_x.shape[3])
            lstm_in_x = lstm_in_x.permute(0,2,1) #temporary
            _, (hidden_state, cell_state) = self.rnn(
                lstm_in_x, (hidden_state, cell_state)
            )
            hidden_state = self.dropout(hidden_state)

        x = hidden_state.squeeze(0)
        for layer_number, dense_layer in enumerate(self.dense_layers):
            x = dense_layer(x)
            if return_last_dense and (layer_number == len(self.dense_layers) - 2):
                output = x
        if return_last_dense:
            return x, output
        return x
