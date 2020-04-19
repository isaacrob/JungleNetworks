import torch
import torch.nn as nn
import torch.nn.functional as F

from jungle_networks import *


class TestDenseJungleNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, internal_dim = 32, n_hidden = 10, jungle_activation = torch.relu):
        super().__init__()

        self.jungle = DenseJungleSubnet(n_inputs, internal_dim, n_hidden = n_hidden, activation = jungle_activation)
        self.out = nn.Linear(internal_dim, n_outputs)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.selu(self.jungle(x))
        x = self.out(x)

        return x

class TestConvolutionalJungleNet(nn.Module):
    def __init__(self, n_outputs, internal_dim = 32, n_hidden = 8, hidden_overlap = None, activation = torch.relu, grow_dims = False):
        super().__init__()

        self.activation = activation
        mult = 2 if grow_dims else 1

        self.jungle1 = ConvolutionalJungleSubnet(1, internal_dim, n_hidden = n_hidden, hidden_overlap = hidden_overlap, activation = activation)
        self.pool1 = nn.MaxPool2d(2)
        self.jungle2 = ConvolutionalJungleSubnet(internal_dim, internal_dim*mult, n_hidden = n_hidden, hidden_overlap = hidden_overlap, activation = activation)
        self.pool2 = nn.MaxPool2d(2)
        self.jungle3 = ConvolutionalJungleSubnet(internal_dim*mult, internal_dim*mult**2, n_hidden = n_hidden, hidden_overlap = hidden_overlap, activation = activation)
        self.pool3 = nn.MaxPool2d(2)
        self.jungle4 = ConvolutionalJungleSubnet(internal_dim*mult**2, internal_dim*mult**3, n_hidden = n_hidden, hidden_overlap = hidden_overlap, activation = activation)
        self.pool4 = nn.AdaptiveAvgPool2d(1)

        self.lin1 = nn.Linear(internal_dim*mult**3, internal_dim*mult**3)

        self.out = nn.Linear(internal_dim*mult**3, n_outputs)

    def forward(self, x):
        # print(x.shape)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        # print(x.shape)

        x = self.pool1(self.jungle1(x))
        x = self.pool2(self.jungle2(x))
        x = self.pool3(self.jungle3(x))
        x = self.pool4(self.jungle4(x))
        # x = F.selu(self.jungle(x))
        x = self.activation(x)
        # print(x.shape)

        x = x.view(x.shape[0], -1)
        x = self.activation(self.lin1(x))
        x = self.out(x)

        return x

class TestConvolutionalForestNet(nn.Module):
    def __init__(self, n_outputs, internal_dim = 32, n_hidden = 3, activation = torch.relu, grow_dims = False):
        super().__init__()

        self.activation = activation
        mult = 2 if grow_dims else 1

        self.jungle1 = ConvolutionalForestSubnet(1, internal_dim, n_hidden = n_hidden, hidden_overlap = None, activation = activation)
        self.pool1 = nn.MaxPool2d(2)
        self.jungle2 = ConvolutionalForestSubnet(internal_dim, internal_dim*mult, n_hidden = n_hidden, hidden_overlap = None, activation = activation)
        self.pool2 = nn.MaxPool2d(2)
        self.jungle3 = ConvolutionalForestSubnet(internal_dim*mult, internal_dim*mult**2, n_hidden = n_hidden, hidden_overlap = None, activation = activation)
        self.pool3 = nn.MaxPool2d(2)
        self.jungle4 = ConvolutionalForestSubnet(internal_dim*mult**2, internal_dim*mult**3, n_hidden = n_hidden, hidden_overlap = None, activation = activation)
        self.pool4 = nn.AdaptiveAvgPool2d(1)

        self.lin1 = nn.Linear(internal_dim*mult**3, internal_dim*mult**3)

        self.out = nn.Linear(internal_dim*mult**3, n_outputs)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])

        x = self.pool1(self.jungle1(x))
        x = self.pool2(self.jungle2(x))
        x = self.pool3(self.jungle3(x))
        x = self.pool4(self.jungle4(x))
        # x = F.selu(self.jungle(x))
        x = self.activation(x)
        # print(x.shape)

        x = x.view(x.shape[0], -1)
        x = self.activation(self.lin1(x))
        x = self.out(x)

        return x

class TestRecurrentClassificationJungleNet(nn.Module):
    def __init__(self, n_outputs, max_len, vocab_size, internal_dim = 32, hidden_overlap = None, activation = torch.relu):
        super().__init__()

        self.activation = activation

        self.embed = nn.Embedding(vocab_size, internal_dim, sparse = True)

        self.jungle1 = RecurrentJungleSubnet(internal_dim, internal_dim, max_len, hidden_overlap, activation)

        self.out = nn.Linear(internal_dim, n_outputs)

    def forward(self, x, offsets):
        embed = self.embed(x)
        _, final_hidden = self.jungle1(embed)
        print(final_hidden.shape)
        final_hidden = self.activation(final_hidden)
        out = self.out(final_hidden)
        print(out.shape)

        return out