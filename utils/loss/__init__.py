import omegaconf
import torch.nn as nn


def get_loss(loss_config, **kwargs):
    if isinstance(loss_config, omegaconf.dictconfig.DictConfig):
        args = loss_config.args
        name = loss_config.name
    elif isinstance(loss_config, dict):
        args = loss_config['args']
        name = loss_config['name']
    else:
        raise RuntimeError(f'Invalid loss_config type: {type(loss_config)}')

    if name == 'CrossEntropy':
        loss = nn.CrossEntropyLoss(**args, **kwargs)  # takes mean by samples
    elif name == 'BCELoss':
        loss = nn.BCELoss(**args, **kwargs)  # takes mean by units
    elif name == 'BCEWithLogitsLoss':
        loss = nn.BCEWithLogitsLoss(**args, **kwargs)
    elif name == 'MSELoss':
        loss = nn.MSELoss(**args, **kwargs)
    elif name == 'NLLLoss':
        loss = nn.NLLLoss(**args, **kwargs)
    else:
        raise RuntimeError(f'No such pre-defined loss: {name}')
    return loss
