from torch import nn, relu
import torch

import math
from pathlib import Path

from .base import ModelBase


class CNN_LSTM_Model(ModelBase):


    """
    A PyTorch replica of the RNN structured model from the original paper. Note that
    this class assumes feature_engineering was run with channels_first=True

    Parameters
    ----------
    in_channels: int, default=9
        Number of channels in the input data. Default taken from the number of bands in the
        MOD09A1 + the number of bands in the MYD11A2 datasets
    num_bins: int, default=32
        Number of bins in the histogram
    hidden_size: int, default=128
        The size of the hidden state. Default taken from the original repository
    rnn_dropout: float, default=0.75
        Default taken from the original paper. Note that this dropout is applied to the
        hidden state after each timestep, not after each layer (since there is only one layer)
    dense_features: list, or None, default=None.
        output feature size of the Linear layers. If None, default values will be taken from the paper.
        The length of the list defines how many linear layers are used.
    savedir: pathlib Path, default=Path('data/models')
        The directory into which the models should be saved.
    device: torch.device
        Device to run model on. By default, checks for a GPU. If none exists, uses
        the CPU
    """
    def __init__(
        self,
        in_channels=9,
        num_bins=32,
        hidden_size=128,
        rnn_dropout=0.75,
        dense_features=None,
        savedir=Path("data/models"),
        use_gp=True,
        sigma=1,
        r_loc=0.5,
        r_year=1.5,
        sigma_e=0.01,
        sigma_b=0.01,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):

        model = CNN_LSTM_NET(
            in_channels=in_channels,
            num_bins=num_bins,
            hidden_size=hidden_size,
            num_rnn_layers=1,
            rnn_dropout=rnn_dropout,
            dense_features=dense_features,
        )

        if dense_features is None:
            num_dense_layers = 2
        else:
            num_dense_layers = len(dense_features)
        model_weight = f"dense_layers.{num_dense_layers - 1}.weight"
        model_bias = f"dense_layers.{num_dense_layers - 1}.bias"

        super().__init__(
            model,
            model_weight,
            model_bias,
            "rnn",
            savedir,
            use_gp,
            sigma,
            r_loc,
            r_year,
            sigma_e,
            sigma_b,
            device,
        )

    def reinitialize_model(self, time=None):
        self.model.initialize_weights()


class CNN_LSTM_NET(nn.Module):
    """
    A crop yield conv net.

    For a description of the parameters, see the RNNModel class.
    """

    def __init__(
        self,
        in_channels=9,
        num_bins=32,
        hidden_size=128,
        num_rnn_layers=1,
        rnn_dropout=0.25,
        dense_features=None,
    ):
        super().__init__()

        if dense_features is None:
            dense_features = [256, 1]
        dense_features.insert(0, hidden_size)

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
                    in_features=dense_features[i - 1], out_features=dense_features[i]
                )
                for i in range(1, len(dense_features))
            ]
        )

        self.device = torch.device('cuda')
        self.conv_blocks = []
        self.batch_blocks = []
        self.dense_blocks = []
        for i in range(36):
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, 32, (2,1)),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.MaxPool2d((2,1)),
                nn.Conv2d(32, 64, (2,1)),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d((2,1)),
            ).to(self.device))

            self.batch_blocks.append(nn.BatchNorm1d(448).to(self.device))
            self.dense_blocks.append(nn.Sequential(
                                    nn.ReLU(),
                                    nn.Linear(256,64),
                                    nn.ReLU()).to(self.device))
                                                
        self.final = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256,1),#64*36, 1),
            nn.ReLU()
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
        #x = x.permute(0, 2, 1, 3).contiguous()
        #x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        x = x.permute(0, 1, 3, 2).contiguous()
        sequence_length = x.shape[3]

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
            input_x = x[:, :, :, i].unsqueeze(3) #input for one lstm cell
            lstm_in_x = self.conv_blocks[i](input_x)
            lstm_in_x = lstm_in_x.view(lstm_in_x.shape[0], lstm_in_x.shape[1] * lstm_in_x.shape[2], lstm_in_x.shape[3])
            input_x = lstm_in_x.permute(0,2,1) #temporary
        
            #input_x = x[:, i, :].unsqueeze(1)
            _, (hidden_state, cell_state) = self.rnn(
                input_x, (hidden_state, cell_state)
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



    """
    A PyTorch replica of the RNN structured model from the original paper. Note that
    this class assumes feature_engineering was run with channels_first=True

    Parameters
    ----------
    in_channels: int, default=9
        Number of channels in the input data. Default taken from the number of bands in the
        MOD09A1 + the number of bands in the MYD11A2 datasets
    num_bins: int, default=32
        Number of bins in the histogram
    hidden_size: int, default=128
        The size of the hidden state. Default taken from the original repository
    rnn_dropout: float, default=0.75
        Default taken from the original paper. Note that this dropout is applied to the
        hidden state after each timestep, not after each layer (since there is only one layer)
    dense_features: list, or None, default=None.
        output feature size of the Linear layers. If None, default values will be taken from the paper.
        The length of the list defines how many linear layers are used.
    savedir: pathlib Path, default=Path('data/models')
        The directory into which the models should be saved.
    device: torch.device
        Device to run model on. By default, checks for a GPU. If none exists, uses
        the CPU

    def __init__(self, in_channels=9, num_bins=32, hidden_size=128, 
                 dense_features=None, savedir=Path('data/models'), use_gp=True,
                 sigma=1, r_loc=0.5, r_year=1.5, sigma_e=0.01, sigma_b=0.01,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

        model = CNN_LSTMNet(in_channels=in_channels, num_bins=num_bins, hidden_size=hidden_size,
                      num_rnn_layers=1,
                      dense_features=dense_features)

        if dense_features is None:
            num_dense_layers = 2
        else:
            num_dense_layers = len(dense_features)
        model_weight = f'dense_layers.{num_dense_layers - 1}.weight'
        model_bias = f'dense_layers.{num_dense_layers - 1}.bias'

        super().__init__(model, model_weight, model_bias, 'rnn', savedir, use_gp, sigma, r_loc, r_year,
                         sigma_e, sigma_b, device)

    def reinitialize_model(self, time=None):
        self.model.initialize_weights()


class CNN_LSTMNet(nn.Module):

    def __init__(self, in_channels=9, num_bins=32, hidden_size=256, num_rnn_layers=1,
                 dense_features=None):
        super().__init__()

        if dense_features is None:
            dense_features = [256, 1]
        dense_features.insert(0, hidden_size)

        self.device = torch.device('cuda')

        self.conv_blocks = []
        self.batch_blocks = []
        self.dense_blocks = []
        for i in range(36):
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, 32, (2,1)),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.MaxPool2d((2,1)),
                nn.Conv2d(32, 64, (2,1)),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d((2,1)),
            ).to(self.device))

            self.batch_blocks.append(nn.BatchNorm1d(448).to(self.device))
            self.dense_blocks.append(nn.Sequential(
                                    nn.ReLU(),
                                    nn.Linear(256,64),
                                    nn.ReLU()).to(self.device))
                                                
        self.final = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256,1),#64*36, 1),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.25)
        self.dropout_2 = nn.Dropout(0.5)

        self.rnn = nn.LSTM(input_size=448,#9*32,
                           hidden_size=hidden_size,
                           num_layers=num_rnn_layers,
                           batch_first=True)
        self.hidden_size = hidden_size

        #Fix initializing and relu layer
        self.dense_layers = nn.ModuleList([
            nn.Linear(in_features=dense_features[i-1],
                      out_features=dense_features[i])
            for i in range(1, len(dense_features))
        ])

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

        # the model expects feature_engineer to have been run with channels_first=True, which means
        # the input is [batch, bands, times, bins].
        # Reshape to [batch, times, bands * bins]
        #x = x.permute(0, 2, 1, 3).contiguous()
        #print(x.shape)
        x = x.permute(0, 1, 3, 2).contiguous()
        #print(x.shape)

        sequence_length = x.shape[3]

        hidden_state = torch.zeros(1, x.shape[0], self.hidden_size)
        cell_state = torch.zeros(1, x.shape[0], self.hidden_size)

        if x.is_cuda:
            hidden_state = hidden_state.cuda()
            cell_state = cell_state.cuda()

        hidden_list = []
        for i in range(sequence_length): #in times
            input_x = x[:, :, :, i].unsqueeze(3) #input for one lstm cell
            #print(input_x.shape)
            #(32,9,32,1)
            #print(input_x.shape)
            lstm_in_x = self.conv_blocks[i](input_x)
            #print(lstm_in_x.shape)
            lstm_in_x = lstm_in_x.view(lstm_in_x.shape[0], lstm_in_x.shape[1] * lstm_in_x.shape[2], lstm_in_x.shape[3])
            #print(lstm_in_x.shape)
            input_x = lstm_in_x.permute(0,2,1) #temporary
            #print(input_x.shape)
            #input_x = self.batch_blocks[i](lstm_in_x).permute(0,2,1)
            #print(input_x)
            #print(input_x.shape)
            _, (hidden_state, cell_state) = self.rnn(input_x,
                                                     (hidden_state, cell_state))
            #print(hidden_state.shape)
            hidden_state = self.dropout(hidden_state) #comment this our for working code
            #hidden_list.append(self.dense_blocks[i](hidden_state))

        #print(hidden_list[0].shape)
        #out = hidden_state.permute(1,0,2)#torch.stack(hidden_list, dim=1).squeeze().permute(1,0,2).contiguous()
        #print(out.shape)
        #exit()
        #print(out.shape)
        #out = out.view(out.shape[0], out.shape[1]*out.shape[2])
        #out = self.dropout_2(out)
        #print(out.shape)
        #x = self.final(out)

        
        print(lstm_in_x.shape)
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        
        sequence_length = x.shape[1]
        for i in range(sequence_length):
            # The reason the RNN is unrolled here is to apply dropout to each timestep;
            # The rnn_dropout argument only applies it after each layer. This better mirrors
            # the behaviour of the Dropout Wrapper used in the original repository
            # https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/DropoutWrapper
            input_x = x[:, i, :].unsqueeze(1)
            #prev 32,1,288
            print(input_x.shape)
            exit()
            _, (hidden_state, cell_state) = self.rnn(input_x,
                                                     (hidden_state, cell_state))
            hidden_state = self.dropout(hidden_state)
        

        
        #comment this out for working code
        x = self.dropout_2(hidden_state.squeeze(0))

        for layer_number, dense_layer in enumerate(self.dense_layers):
            x = dense_layer(x)
            if return_last_dense and (layer_number == len(self.dense_layers) - 2):
                output = x
        if return_last_dense:
            return x, output
        
        return x
    """