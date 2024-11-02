import os
from collections import defaultdict

import numpy as np
import pytorch_lightning as L
import torch
from tqdm.notebook import tqdm

from .trainer_utils import get_pl_trainer
from ..config import dict_to_config


def merge_batch_prediction(predictions, keys=None, exclude_keys=None):
    merge_preds = defaultdict(lambda: {'values': [], 'contains_none': False, 'dtype': None})
    keys = keys if keys is not None else predictions[0].keys()
    exclude_keys = exclude_keys if exclude_keys is not None else []

    for batch in predictions:
        for key, value in batch.items():
            if value is None:
                merge_preds[key]['contains_none'] = True
            else:
                if isinstance(value, torch.Tensor):
                    merge_preds[key]['values'].append(value)
                    merge_preds[key]['dtype'] = torch.Tensor
                elif isinstance(value, (list, tuple)):
                    merge_preds[key]['values'].extend(value)
                    merge_preds[key]['dtype'] = type(value)
                else:
                    raise ValueError(f'Unsupported type: {type(value)}')

    for key, data in merge_preds.items():
        if key in keys and key not in exclude_keys:
            # print(f'[{key}]:', data['dtype'], len(data['values']), 'batches')
            if data['contains_none']:
                merge_preds[key] = None
            else:
                if data['dtype'] == torch.Tensor:
                    try:
                        merge_preds[key] = torch.cat(data['values'], dim=0)
                    except RuntimeError as e:
                        print(f'[Warning]: Cannot concatenate tensor along dimension 0 for key {key}')
                elif data['dtype'] in (list, tuple):
                    merge_preds[key] = np.array(data['values'])
                else:
                    raise ValueError(f'Unsupported type: {type(data["values"])}')
        else:
            merge_preds[key] = data['values']
    return merge_preds


def get_ckpt(log_dir, version, epoch):
    if type(version) == int:
        version = f'version_{version}'

    if type(epoch) == str and 'last' in epoch:
        string = epoch
    else:
        if type(epoch) == int and epoch < 10:
            epoch = f'0{epoch}'
        string = f'epoch={epoch}' if 'epoch' not in epoch else epoch

    args_dir = log_dir + f'/{version}/'
    ckpt_dir = log_dir + f'/{version}/checkpoints/'
    files = os.listdir(ckpt_dir)
    for file in files:
        if string in file:
            return ckpt_dir + file, args_dir + 'hparams.yaml'


class LitModelInference:
    def __init__(self, model_class, data_module_class, ckpt_path=None, device=None, log=True, use_trainer=False):
        self.ckpt_path = ckpt_path
        self.log = log
        self.use_trainer = use_trainer

        if self.log:
            print('[loading checkpoint]:', ckpt_path)

        if isinstance(device, str):
            if device is None or device == 'auto':
                map_location = None
                accelerator = 'auto'
            elif 'cuda' in device:
                # use the default gpu if device == 'cuda'
                map_location = device  # use the specific gpu
                accelerator = 'gpu'
            elif device == 'cpu':
                map_location = 'cpu'
                accelerator = 'cpu'
            else:
                raise ValueError(f'No such pre-defined device {device}')
        elif isinstance(device, torch.device):
            map_location = 'gpu' if device.type == 'cuda' else 'cpu'
            accelerator = 'gpu' if device.type == 'cuda' else 'cpu'
        else:
            map_location = None
            accelerator = 'auto'

        self.ckpt_config = \
        dict_to_config(torch.load(self.ckpt_path, weights_only=False, map_location=map_location)['hyper_parameters'])[
            'config']
        self.ckpt_model = model_class.load_from_checkpoint(ckpt_path, map_location=map_location)
        self.device = next(self.ckpt_model.parameters()).device  # auto-detect the used device
        self.map_location = map_location
        self.accelerator = accelerator
        self.ckpt_config.trainer.accelerator = accelerator
        self.ckpt_config.trainer.devices = 10 if str(self.device) == 'cpu' else [int(self.device.index)]
        self.data_module_class = data_module_class
        self.pl_data_module = data_module_class(self.ckpt_config, log=log)
        self.ckpt_trainer = get_pl_trainer(self.ckpt_config)
        self.ckpt_trainer.logger = None

        L.seed_everything(seed=self.ckpt_config.seed, workers=True)

    def set_batch_size(self, batch_size=None, num_workers=1):
        if batch_size is None:
            batch_size = self.ckpt_config.predict_dataloader.batch_size
        self.ckpt_config.predict_dataloader.batch_size = batch_size
        self.ckpt_config.predict_dataloader.num_workers = num_workers
        self.ckpt_config.predict_dataloader.pin_memory = False
        self.ckpt_config.predict_dataloader.persistent_workers = False
        # torch.multiprocessing.set_sharing_strategy('file_system')

    def predict(self, sequence=None, dataset=None, subset=None, **kwargs):
        if sum([sequence is not None, dataset is not None, subset is not None]) != 1:
            raise ValueError('Please set one of the parameters: sequences, dataset, subset')

        self.pl_data_module.prepare_predict_data(sequence=sequence, dataset=dataset, subset=subset, **kwargs)
        self.pl_data_module.setup('predict')

        if self.use_trainer:
            predictions = self.ckpt_trainer.predict(
                model=self.ckpt_model,
                dataloaders=self.pl_data_module.predict_dataloader(),
                ckpt_path=self.ckpt_path
            )
        else:
            predictions = []
            progress_bar = kwargs.get('progress_bar', True) and self.log
            dataloader = self.pl_data_module.predict_dataloader()
            dataloader = tqdm(enumerate(dataloader), total=len(dataloader)) if progress_bar else enumerate(dataloader)
            self.ckpt_model.eval()
            with torch.no_grad():
                for i, batch in dataloader:
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                    predictions.append(self.ckpt_model.predict_step(batch, i))
        return predictions
