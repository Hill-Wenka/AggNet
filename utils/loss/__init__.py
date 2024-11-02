import omegaconf
import torch.nn as nn

from .BinaryClassificationLoss import FocalLoss, MutualInformationLoss, WeightedLoss
from .ContrastiveLoss import ContrastiveRankLoss, NTXentLoss, SimCSELoss, SupConLoss
from .MMDLoss import MMDLoss


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
    elif name == 'FocalLoss':
        loss = FocalLoss(**args, **kwargs)
    elif name == 'WeightedLoss':
        loss = WeightedLoss(**args, **kwargs)
    elif name == 'NTXentLoss':
        loss = NTXentLoss(**args, **kwargs)
    elif name == 'SupConLoss':
        loss = SupConLoss(**args, **kwargs)
    elif name == 'SimCSELoss':
        loss = SimCSELoss(**args, **kwargs)
    elif name == 'MutualInformationLoss':
        loss = MutualInformationLoss(**args, **kwargs)
    elif name == 'ContrastiveRankLoss':
        loss = ContrastiveRankLoss(**args, **kwargs)
    elif name == 'MSELoss':
        loss = nn.MSELoss(**args, **kwargs)
    elif name == 'NLLLoss':
        loss = nn.NLLLoss(**args, **kwargs)
    elif name == 'MMDLoss':
        loss = MMDLoss(**args, **kwargs)
    else:
        raise RuntimeError(f'No such pre-defined loss: {name}')
    return loss
