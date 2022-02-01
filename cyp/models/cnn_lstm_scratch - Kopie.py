from torch import nn, relu
import torch
import torch.nn.functional as F

import math
from pathlib import Path

from .base import ModelBase


class CNN_LSTM_scratchModel(ModelBase):
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

        model = CNN_LSTM_SCRATCH_NET(
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
        pass


class CNN_LSTM_SCRATCH_NET(nn.Module):
    def __init__(self, in_channels, num_bins, hidden_size, num_rnn_layers, rnn_dropout, dense_features):
        super(CNN_LSTM_SCRATCH_NET, self).__init__()

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

        self.lstm = nn.LSTM(input_size=448, hidden_size=256, num_layers=1)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(36*64, 1)
       
    def forward(self, x_3d):
        hidden = None
        out_list = []
        #input is [batch, bands, times, bins]
        # we need [batch, times, bands, bins, 1]
        x_3d = x_3d.permute(0, 2, 1, 3)
        x_3d = torch.unsqueeze(x_3d, dim=-1)
        for t in range(x_3d.size(1)):
            x = self.conv_blocks(x_3d[:, t, :, :, :])
            lstm_in_x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
            x = lstm_in_x.permute(0,2,1) #temporary
            out, hidden = self.lstm(x, hidden)
            out_list.append(out)
            #out, hidden = self.lstm(x.unsqueeze(0), hidden)         

        out = torch.stack(out_list, dim=1).squeeze().permute(1,0,2).contiguous()
        #print(out.shape)
        out = out.view(out.shape[0], out.shape[1]*out.shape[2])
        x = self.fc1(out)
        x = F.relu(x)
        x = self.fc2(x)
        return x