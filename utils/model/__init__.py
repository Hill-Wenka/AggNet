import torch.nn as nn

activations = {'ReLU': nn.ReLU,
               'Sigmoid': nn.Sigmoid,
               'LeakyReLU': nn.LeakyReLU,
               'Tanh': nn.Tanh,
               'ELU': nn.ELU,
               'SELU': nn.SELU,
               'Softplus': nn.Softplus,
               'SiLU': nn.SiLU,
               'Swish': nn.SiLU,
               'Hardswish': nn.Hardswish,
               'GELU': nn.GELU,
               'Mish': nn.Mish}

poolings = {'max_pooling': nn.MaxPool1d,
            'avg_pooling': nn.AvgPool1d,
            'max_pooling2d': nn.MaxPool2d,
            'avg_pooling2d': nn.AvgPool2d}
