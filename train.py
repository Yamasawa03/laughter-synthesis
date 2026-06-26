import os
import csv
import pytorch_lightning as pl
import hydra
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from data_module import DataModule
from lightning_module import BaselineLightningModule

seed = 1024
seed_everything(seed)

@hydra.main(config_path='config', config_name='default')
def train(cfg):
    # loggers
    csvlogger = CSVLogger(save_dir=cfg.log_dir, name='csv')
    loggers = [csvlogger]
    if cfg.use_tb:
        tblogger = TensorBoardLogger(save_dir=cfg.log_dir, name='tb')
        loggers.insert(0, tblogger)

    # callbacks
    every_n_epochs = cfg.get('every_n_epochs', 50)
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.log_dir,
                            save_top_k=-1, save_last=True,
                            every_n_epochs=every_n_epochs, monitor='val_loss', mode='min')
    
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.log_dir,
        save_top_k=1,
        filename='best-{epoch}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min'
    )

    lr_monitor = LearningRateMonitor()
    callbacks = [checkpoint_callback, best_checkpoint_callback, lr_monitor]
    if cfg.train.early_stop:
        earlystop_callback = EarlyStopping(monitor='val_loss', min_delta=1e-3,
                                patience=10, mode='min', check_finite=True,
                                stopping_threshold=0.0, divergence_threshold=1e5)
        callbacks.append(earlystop_callback)

    datamodule = DataModule(cfg)
    lightning_module = BaselineLightningModule(cfg)
    trainer = pl.Trainer(
        **cfg.train.trainer,
        logger=loggers,
        callbacks=callbacks,
        limit_train_batches=1.0 if not cfg.debug else 0.1,
        limit_val_batches=1.0 if not cfg.debug else 0.5)
    ckpt_path = cfg.get('ckpt_path', None)
    strict_load = cfg.get('strict_load', True)

    if ckpt_path and not strict_load:
        import torch
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        model_state = lightning_module.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in model_state and v.shape == model_state[k].shape}
        skipped = sorted(set(state_dict.keys()) - set(filtered.keys()))
        lightning_module.load_state_dict(filtered, strict=False)
        print(f"Cross-config resume: loaded {len(filtered)}/{len(state_dict)} keys")
        if skipped:
            print(f"Skipped keys ({len(skipped)}): {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
        trainer.fit(lightning_module, datamodule=datamodule)
    else:
        trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)
    print(f'Training ends, best score: {best_checkpoint_callback.best_model_score}, ckpt path: {best_checkpoint_callback.best_model_path}')
    if cfg.train.run_test_after_fit:
        trainer.test(lightning_module, datamodule=datamodule)

if __name__ == '__main__':
    train()
