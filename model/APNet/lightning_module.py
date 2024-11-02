import lightning as L
import omegaconf
import pandas as pd
import torch

from model.APNet import APNet
from utils.metric.bin_cls import compute_metrics
from utils.optim import get_optimizer, get_scheduler


def save_metrics(epoch, step, stage, metrics, save_path, old_df=None):
    metrics = metrics.copy()
    keys = list(metrics.keys())
    metrics['epoch'] = epoch
    metrics['step'] = step
    metrics['stage'] = stage
    metrics['loss'] = round(metrics['loss'], 4)
    columns = ['epoch', 'step', 'stage'] + keys
    df = pd.DataFrame([metrics], columns=columns)
    if old_df is not None:
        df = pd.concat([old_df, df]).reset_index(drop=True)
    df.to_csv(save_path + '/metrics.csv', index=False)
    return df


class LightningModule(L.LightningModule):
    def __init__(self, config: omegaconf.dictconfig.DictConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.configure_model()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.metrics = None

    def configure_model(self):
        self.model = APNet.APNet(**self.config.hparams)

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = get_optimizer(self.config.optimizer, params)
        optimizers = [optimizer]
        scheduler = get_scheduler(self.config.scheduler, optimizer)
        schedulers = [scheduler]

        if scheduler is None:
            return optimizers
        else:
            return optimizers, schedulers

    def compute_loss(self, labels, batch_outputs):
        preds = batch_outputs['cls_head']
        loss = self.model.compute_loss(preds, labels)
        return loss

    def batch_forward(self, batch):
        batch_labels, batch_seqs = batch['batch_labels'], batch['batch_seqs']
        batch_tokens, batch_embedding = batch['batch_tokens'], batch['batch_embeddings']
        batch_aaindex = batch['batch_aaindex']
        x = batch_embedding if batch_embedding is not None else batch_tokens
        batch_outputs = self.model(x, features=batch_aaindex, add_special_tokens=False)
        loss = self.compute_loss(batch_labels, batch_outputs)
        return {'loss': loss, 'labels': batch_labels, 'preds': batch_outputs['cls_head']}

    def compute_batch_metrics(self, batch_outputs):
        batch_metrics = {}
        for key, value in batch_outputs.items():
            if 'loss' in key:
                batch_metrics[key] = value
        return batch_metrics

    def compute_epoch_metrics(self, epoch_outputs, stage):
        epoch_metrics = {}
        epoch_metrics['loss'] = torch.tensor(
            [batch['loss'] for batch in epoch_outputs if not torch.isnan(batch['loss'])]).mean().detach().item()

        preds = torch.cat([batch['preds'] for batch in epoch_outputs], dim=0).squeeze()
        labels = torch.cat([batch['labels'] for batch in epoch_outputs], dim=0).squeeze()
        not_nan_idx = ~torch.isnan(labels).cpu().numpy()
        not_nan_preds = preds[not_nan_idx].detach().cpu().numpy()
        not_nan_labels = labels[not_nan_idx].detach().cpu().numpy()
        metrics = compute_metrics(not_nan_preds, not_nan_labels, softmax=True, only_dict=True)
        epoch_metrics.update(metrics)
        self.metrics = save_metrics(self.current_epoch,
                                    self.global_step,
                                    stage,
                                    epoch_metrics,
                                    self.logger.log_dir,
                                    self.metrics)
        return epoch_metrics

    def training_step(self, batch, batch_idx):
        return self.batch_forward(batch)

    def validation_step(self, batch, batch_idx):
        return self.batch_forward(batch)

    def test_step(self, batch, batch_idx):
        return self.batch_forward(batch)

    def on_train_batch_end(self, batch_outputs, batch, batch_idx, **kwargs):
        batch_metrics = self.compute_batch_metrics(batch_outputs)
        batch_metrics = {'train/' + k + '_step': v for k, v in batch_metrics.items()}
        self.log_dict(batch_metrics, on_step=True, on_epoch=False)
        self.training_step_outputs.append(batch_outputs)

    def on_validation_batch_end(self, batch_outputs, batch, batch_idx, **kwargs):
        batch_metrics = self.compute_batch_metrics(batch_outputs)
        batch_metrics = {'valid/' + k + '_step': v for k, v in batch_metrics.items()}
        self.log_dict(batch_metrics, on_step=True, on_epoch=False)
        self.validation_step_outputs.append(batch_outputs)

    def on_test_batch_end(self, batch_outputs, batch, batch_idx, **kwargs):
        batch_metrics = self.compute_batch_metrics(batch_outputs)
        batch_metrics = {'test/' + k + '_step': v for k, v in batch_metrics.items()}
        self.log_dict(batch_metrics, on_step=True, on_epoch=False)
        self.test_step_outputs.append(batch_outputs)

    def on_train_epoch_end(self):
        epoch_metrics = self.compute_epoch_metrics(self.training_step_outputs, 'train')
        epoch_metrics = {'train/' + k + '_epoch': v for k, v in epoch_metrics.items()}
        self.log_dict(epoch_metrics, on_step=False, on_epoch=True)
        self.training_step_outputs = []

    def on_validation_epoch_end(self):
        epoch_metrics = self.compute_epoch_metrics(self.validation_step_outputs, 'valid')
        epoch_metrics = {'valid/' + k + '_epoch': v for k, v in epoch_metrics.items()}
        self.log_dict(epoch_metrics, on_step=False, on_epoch=True)
        self.validation_step_outputs = []

    def on_test_epoch_end(self):
        epoch_metrics = self.compute_epoch_metrics(self.test_step_outputs, 'test')
        epoch_metrics = {'test/' + k + '_epoch': v for k, v in epoch_metrics.items()}
        self.log_dict(epoch_metrics, on_step=False, on_epoch=True)
        self.test_step_outputs = []

    def predict_step(self, batch, batch_idx, **kwargs):
        batch_labels, batch_seqs = batch['batch_labels'], batch['batch_seqs']
        batch_tokens, batch_embeddings = batch['batch_tokens'], batch['batch_embeddings']
        batch_aaindex = batch['batch_aaindex']
        x = batch_embeddings if batch_embeddings is not None else batch_tokens
        batch_outputs = self.model(x, features=batch_aaindex, add_special_tokens=False)
        loss = self.compute_loss(batch_labels, batch_outputs)
        return {'loss': loss,
                'preds': batch_outputs['cls_head'],
                'latent': batch_outputs['latent'],
                'labels': batch_labels,
                'sequences': batch_seqs,
                'tokens': batch_tokens,
                'embeddings': batch_embeddings}
