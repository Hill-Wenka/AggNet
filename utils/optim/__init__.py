from torch.optim import Adagrad, Adam, AdamW, RAdam, RMSprop, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ExponentialLR, LinearLR


def get_optimizer(optimizer_config, params):
    if optimizer_config.name == 'Adam':
        optimizer = Adam(params, **optimizer_config.args)
    elif optimizer_config.name == 'SGD':
        optimizer = SGD(params, **optimizer_config.args)
    elif optimizer_config.name == 'Adagrad':
        optimizer = Adagrad(params, **optimizer_config.args)
    elif optimizer_config.name == 'RMSprop':
        optimizer = RMSprop(params, **optimizer_config.args)
    elif optimizer_config.name == 'AdamW':
        optimizer = AdamW(params, **optimizer_config.args)
    elif optimizer_config.name == 'RAdam':
        optimizer = RAdam(params, **optimizer_config.args)
    else:
        raise RuntimeError(f'No such pre-defined optimizer: {optimizer_config.name}')
    return optimizer


def get_scheduler(scheduler_config, optimizer):
    if scheduler_config is None or scheduler_config.name is None or scheduler_config.name == '':
        return None
    if scheduler_config.name == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, **scheduler_config.args)
    elif scheduler_config.name == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, **scheduler_config.args)
    elif scheduler_config.name == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, **scheduler_config.args)
    elif scheduler_config.name == 'LinearLR':
        scheduler = LinearLR(optimizer, **scheduler_config.args)
    else:
        raise RuntimeError(f'No such pre-defined scheduler: {scheduler_config.name}')
    scheduler = {'scheduler': scheduler, 'interval': scheduler_config.interval, 'frequency': scheduler_config.frequency}
    return scheduler
