import torch.nn as nn

from . import activations
from ..datastructure.json_utils import json2list


class MLP(nn.Module):
    def __init__(self, hiddens, activation='ReLU', bias=True, batch_norm=False, layer_norm=False, dropout=None,
                 final_transform=None):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(hiddens) - 1):
            if i < len(hiddens) - 2:
                layers.append(nn.Linear(hiddens[i], hiddens[i + 1], bias=not batch_norm and bias))
                if batch_norm:  # only for [N, D] input
                    layers.append(nn.BatchNorm1d(hiddens[i + 1]))
                if layer_norm:  # only for [N, L, D] input
                    layers.append(nn.LayerNorm(hiddens[i + 1]))
                layers.append(activations[activation]())
                if dropout is not None and dropout > 0:
                    layers.append(nn.Dropout(dropout))
            else:
                layers.append(nn.Linear(hiddens[i], hiddens[i + 1], bias=bias))

        self.hiddens = hiddens
        self.activation = activation
        self.bias = bias
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.mlp = nn.Sequential(*layers)
        self.final_transform = activations[final_transform]() if isinstance(final_transform, str) else final_transform

    def forward(self, x):
        x = self.mlp(x)
        if self.final_transform is not None:
            x = self.final_transform(x)
        return x


def get_hiddens(hiddens, input_dim=None, output_dim=None):
    hiddens = json2list(hiddens)
    hiddens = [eval(str(hidden)) for hidden in hiddens]
    input_dim = eval(str(input_dim))
    output_dim = eval(str(output_dim))

    if input_dim:
        if hiddens[0] == -1:
            hiddens = hiddens[1:] if hiddens[1] == input_dim else [input_dim] + hiddens[1:]
        else:
            if hiddens[0] != input_dim:
                raise ValueError(f'Assert {hiddens[0]} == {input_dim}')
    if output_dim:
        if hiddens[-1] == -1:
            hiddens = hiddens[:-1] if hiddens[-2] == output_dim else hiddens[:-1] + [output_dim]
        else:
            if hiddens[-1] != output_dim:
                raise ValueError(f'Assert {hiddens[-1]} == {output_dim}')
    return hiddens
