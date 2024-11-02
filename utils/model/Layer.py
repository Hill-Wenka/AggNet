import torch
import torch.nn as nn

from . import activations
from ..datastructure.json_utils import json2list


class Embedding(nn.Module):
    def __init__(
            self,
            vocab_size,
            dim_embedding,
            custom_embedding=None,
            freeze_embedding=False,
    ):
        super(Embedding, self).__init__()

        self.embedding = nn.Embedding(vocab_size, dim_embedding)
        if custom_embedding is not None:
            self.embedding = self.embedding.from_pretrained(custom_embedding, freeze=freeze_embedding)

    def forward(self, x):
        return self.embedding(x)  # (N, L, D)


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


class TaskHead(nn.Module):
    def __init__(self, hiddens, activation='ReLU', batch_norm=False, bias=True, dropout=None, final_activation=None):
        super(TaskHead, self).__init__()
        self.mlp = MLP(hiddens, activation, bias, batch_norm, dropout)
        self.final_activation = final_activation() if final_activation is not None else None

    def forward(self, x):
        x = self.mlp(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x


class GlobalAveragePooling(nn.Module):
    def __init__(self, dim, size):
        super().__init__()
        assert dim == 1 or dim == 2
        self.dim = dim
        self.size = size
        if self.dim == 2:
            self.pool = nn.AdaptiveAvgPool2d(size)
        elif self.dim == 1:
            self.pool = nn.AdaptiveAvgPool1d(size)

    def forward(self, x):
        # x: [N, C, H, W] for AdaptiveAvgPool2d
        # x: [N, C, L] for AdaptiveAvgPool1d
        output = self.pool(x)  # x: [N, C, D=self.size]
        return output


class GlobalMaxPooling(nn.Module):
    def __init__(self, dim, size):
        super().__init__()
        assert dim == 1 or dim == 2
        self.dim = dim
        self.size = size
        if self.dim == 2:
            self.pool = nn.AdaptiveMaxPool2d(size)
        elif self.dim == 1:
            self.pool = nn.AdaptiveMaxPool1d(size)

    def forward(self, x):
        # x: [N, C, H, W] for AdaptiveAvgPool2d
        # x: [N, C, L] for AdaptiveAvgPool1d
        output = self.pool(x)  # x: [N, C, D=self.size]
        return output


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


def sequence_embedding_aggregation(embeddings, aggregation='mean', remove_cls_eos_token=True):
    # embeddings: (N, L, D)
    cls_token = embeddings[:, 0]  # (N, D)
    embeddings = embeddings[:, 1:-1] if remove_cls_eos_token else embeddings  # (N, L, D)

    if aggregation == 'concat':
        agg_embed = embeddings.reshape(embeddings.shape[0], -1)  # (N, L*D)
    elif aggregation == 'mean':
        agg_embed = GlobalAveragePooling(dim=1, size=1).average_pooling(embeddings.transpose(-1, -2)).transpose(-1,
                                                                                                                -2).squeeze(
            -2)  # (N, D)
    elif aggregation == 'max':
        agg_embed = GlobalMaxPooling(dim=1, size=1)(embeddings.transpose(-1, -2)).transpose(-1, -2).squeeze(
            -2)  # (N, D)
    elif aggregation == 'cls':
        agg_embed = cls_token  # (N, D)
    elif aggregation == 'mean_max':
        agg_embed = torch.cat(
            [GlobalAveragePooling(dim=1, size=1)(embeddings.transpose(-1, -2)).transpose(-1, -2).squeeze(-2),
             GlobalMaxPooling(dim=1, size=1)(embeddings.transpose(-1, -2)).transpose(-1, -2).squeeze(-2)], dim=-1)
    elif aggregation == 'mean_cls':
        agg_embed = torch.cat(
            [GlobalAveragePooling(dim=1, size=1)(embeddings.transpose(-1, -2)).transpose(-1, -2).squeeze(-2),
             cls_token], dim=-1)
    elif aggregation == 'max_cls':
        agg_embed = torch.cat(
            [GlobalMaxPooling(dim=1, size=1)(embeddings.transpose(-1, -2)).transpose(-1, -2).squeeze(-2),
             cls_token], dim=-1)
    elif aggregation == 'mean_max_cls':
        agg_embed = torch.cat(
            [GlobalAveragePooling(dim=1, size=1)(embeddings.transpose(-1, -2)).transpose(-1, -2).squeeze(-2),
             GlobalMaxPooling(dim=1, size=1)(embeddings.transpose(-1, -2)).transpose(-1, -2).squeeze(-2),
             cls_token], dim=-1)
    else:
        raise NotImplementedError
    return agg_embed


def get_aggregation_dim(aggregation, embed_dim, max_len=6):
    if aggregation == 'concat':
        feature_dim = max_len * embed_dim
    elif aggregation in ['max', 'mean', 'cls']:
        feature_dim = embed_dim
    elif aggregation in ['mean_max', 'max_cls', 'mean_cls']:
        feature_dim = embed_dim * 2
    elif aggregation == 'mean_max_cls':
        feature_dim = embed_dim * 3
    else:
        raise NotImplementedError
    return feature_dim


if __name__ == '__main__':
    a = torch.tensor(
        [[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], [[2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]],
        dtype=torch.float32
    )
    print(a.shape)
    print(a)
    print(a.transpose(-1, -2))
    avg_pool_1d = GlobalAveragePooling(dim=1, size=2)
    max_pool_1d = GlobalMaxPooling(dim=1, size=2)
    b = avg_pool_1d(a.transpose(-1, -2)).transpose(-1, -2)
    c = max_pool_1d(a.transpose(-1, -2)).transpose(-1, -2)
    print(b)
    print(b.size())
    print(c)
    print(c.size())
