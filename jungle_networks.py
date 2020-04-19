import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class DenseJungleSubnet(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_hidden = 10, activation = torch.relu):
        super().__init__()

        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.activation = activation

        self.jungle = nn.ModuleList()
        for i in range(n_hidden):
            new_layer = nn.Linear(n_inputs + i, n_outputs + n_hidden - i)
            self.jungle.append(new_layer)

    def forward(self, x):
        internal_state = torch.zeros(x.shape[0], self.n_outputs + self.n_hidden + self.n_inputs).to(device)

        # print(x.shape)
        # print(self.n_inputs)
        # print(internal_state.shape)
        internal_state[:, :self.n_inputs] = x
        for i, layer in enumerate(self.jungle):
            layer_in = internal_state[:, :self.n_inputs + i].clone()
            # print(layer_in.shape)
            layer_out = layer(layer_in)
            layer_out = self.activation(layer_out)
            internal_state[:, self.n_inputs + i:] = layer_out + internal_state[:, self.n_inputs + i:]

        output = internal_state[:, -self.n_outputs:]

        return output

class ConvolutionalJungleSubnet(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_hidden = 8, hidden_overlap = None, activation = torch.relu):
        super().__init__()

        if hidden_overlap is None:
            hidden_overlap = n_outputs

        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.activation = activation
        self.hidden_overlap = hidden_overlap
        self.total_size = n_inputs + n_hidden*hidden_overlap + n_outputs

        self.jungle = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_hidden):
            this_layer_size = n_inputs + i*hidden_overlap
            new_layer = nn.Conv2d(this_layer_size, self.total_size - this_layer_size, 3, 1, 1)
            self.jungle.append(new_layer)
            new_bn = nn.BatchNorm2d(self.total_size - this_layer_size)
            self.bns.append(new_bn)

    def forward(self, x):
        internal_state = torch.zeros(x.shape[0], self.total_size, x.shape[2], x.shape[3]).to(device)

        # print(x.shape)
        # print(self.n_inputs)
        # print(internal_state.shape)
        internal_state[:, :self.n_inputs] = x
        for i, layer in enumerate(self.jungle):
            this_layer_size = self.n_inputs + i*self.hidden_overlap
            layer_in = internal_state[:, :this_layer_size].clone()
            # print(layer_in.shape)
            layer_out = layer(layer_in)
            layer_out = self.bns[i](self.activation(layer_out))
            # print(layer_out.shape)
            # print(internal_state[:, this_layer_size:].shape)
            # print(self.total_size)
            internal_state[:, this_layer_size:] = layer_out + internal_state[:, this_layer_size:]

        output = internal_state[:, -self.n_outputs:]

        return output

class ConvolutionalForestSubnet(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_hidden = 10, hidden_overlap = None, activation = torch.relu):
        super().__init__()

        if hidden_overlap is None:
            hidden_overlap = n_outputs

        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.activation = activation
        self.hidden_overlap = hidden_overlap
        self.total_size = n_inputs + n_hidden*hidden_overlap + n_outputs

        self.jungle = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_hidden):
            this_layer_size = n_inputs + i*hidden_overlap
            new_layer = nn.Conv2d(this_layer_size, n_outputs, 3, 1, 1)
            self.jungle.append(new_layer)
            new_bn = nn.BatchNorm2d(n_outputs)
            self.bns.append(new_bn)

    def forward(self, x):
        internal_state = torch.zeros(x.shape[0], self.total_size, x.shape[2], x.shape[3]).to(device)

        # print(x.shape)
        # print(self.n_inputs)
        # print(internal_state.shape)
        internal_state[:, :self.n_inputs] = x
        for i, layer in enumerate(self.jungle):
            this_layer_size = self.n_inputs + i*self.hidden_overlap
            layer_in = internal_state[:, :this_layer_size].clone()
            # print(layer_in.shape)
            layer_out = layer(layer_in)
            layer_out = self.bns[i](self.activation(layer_out))
            # print(layer_out.shape)
            # print(self.n_hidden - i + 1)
            layer_out = layer_out.repeat(1, self.n_hidden - i + 1, 1, 1)
            # print(layer_out.shape)
            # print(internal_state[:, this_layer_size:].shape)
            # print(self.total_size)
            internal_state[:, this_layer_size:] = layer_out + internal_state[:, this_layer_size:]

        output = internal_state[:, -self.n_outputs:]

        return output

class RecurrentJungleSubnet(nn.Module):
    def __init__(self, n_inputs, n_outputs, max_length = 10, hidden_overlap = None, activation = torch.relu):
        super().__init__()

        if hidden_overlap is None:
            hidden_overlap = n_outputs

        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.n_hidden = max_length
        self.activation = activation
        self.hidden_overlap = hidden_overlap
        self.total_size = n_inputs + max_length*hidden_overlap + n_outputs

        self.jungle = nn.ModuleList()
        for i in range(max_length):
            this_layer_size = n_inputs + i*hidden_overlap
            new_layer = nn.Linear(this_layer_size, self.total_size - this_layer_size)
            self.jungle.append(new_layer)

    def forward(self, x):
        internal_state = torch.zeros(x.shape[1], self.total_size).to(device)
        outputs = torch.zeros(x.shape[0], x.shape[1], self.n_outputs)

        # print(x.shape)
        # print(self.n_inputs)
        # print(internal_state.shape)
        for i, layer in enumerate(self.jungle):
            if i >= x.shape[0]:
                break

            internal_state[:, :self.n_inputs] = x[i]
            this_layer_size = self.n_inputs + i*self.hidden_overlap
            layer_in = internal_state[:, :this_layer_size].clone()
            # print(layer_in.shape)
            layer_out = layer(layer_in)
            layer_out = self.activation(layer_out)
            # print(layer_out.shape)
            # print(internal_state[:, this_layer_size:].shape)
            # print(self.total_size)
            internal_state[:, this_layer_size:] = layer_out + internal_state[:, this_layer_size:]
            outputs[i] = internal_state[:, -self.n_outputs:]

        final_output = outputs[-1]

        return outputs, final_output