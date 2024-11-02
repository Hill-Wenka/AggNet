import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from tqdm import tqdm


class LitProgressBar(TQDMProgressBar):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_custom_tqdm = kwargs.get('use_custom_tqdm', False)

    def init_sanity_tqdm(self):
        if self.use_custom_tqdm:
            return tqdm(desc='Sanity Checking', leave=False)
        else:
            return super().init_sanity_tqdm()

    def init_train_tqdm(self):
        if self.use_custom_tqdm:
            return tqdm(desc='Training', leave=False)
        else:
            return super().init_train_tqdm()

    def init_validation_tqdm(self):
        if self.use_custom_tqdm:
            return tqdm(desc='Validating', leave=False)
        else:
            return super().init_validation_tqdm()

    def init_test_tqdm(self):
        if self.use_custom_tqdm:
            return tqdm(desc='Testing', leave=False)
        else:
            return super().init_test_tqdm()

    def init_predict_tqdm(self):
        if self.use_custom_tqdm:
            return tqdm(desc='Predicting', leave=False)
        else:
            return super().init_predict_tqdm()


def get_pl_trainer(config) -> L.Trainer:
    progress_bar_callback = LitProgressBar(**config.progress_bar_callback)
    logger = None if config.logger.save_dir is None else [TensorBoardLogger(**config.logger)]
    checkpoint_callback = None if config.ckpt_callback is None else ModelCheckpoint(**config.ckpt_callback)
    early_stop_callback = None if config.early_stop_callback is None else EarlyStopping(**config.early_stop_callback)
    swa_callback = None if config.swa_callback is None or config.swa_callback == '' else StochasticWeightAveraging(
        **config.swa_callback
    )

    callbacks = [checkpoint_callback, early_stop_callback, progress_bar_callback, swa_callback]
    callbacks = [x for x in callbacks if x is not None]

    if config.trainer.strategy is not None and 'ddp' in config.trainer.strategy:
        ddp_strategy = DDPStrategy(find_unused_parameters=False)
        trainer = L.Trainer(
            logger=logger,
            callbacks=callbacks,
            strategy=ddp_strategy,
            **config.trainer
        )
    else:
        trainer = L.Trainer(
            logger=logger,
            callbacks=callbacks,
            **config.trainer
        )
    return trainer
