import os
from pathlib import Path

from pprint import pformat
import logging # we use top level logging since most actual diagnostic info is in libs
import hydra
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)

from pytorch_lightning.loggers import WandbLogger
import wandb

from config import RootConfig, Metric
from data import SpikingDataset
from model import BrainBertInterface
from utils import get_latest_ckpt_from_wandb_id

@hydra.main(version_base=None, config_path='config', config_name="config")
def run_exp(cfg : RootConfig) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Running NDT2, dumping config:")
    logger.info(OmegaConf.to_yaml(cfg))
    pl.seed_everything(seed=cfg.seed)

    dataset = SpikingDataset(cfg.dataset)
    train, val = dataset.create_tv_datasets()
    logger.info(f"Training on {len(train)} examples")

    data_attrs = dataset.get_data_attrs()
    logger.info(pformat(f"Data attributes: {data_attrs}"))
    if cfg.init_from_id:
        init_ckpt = get_latest_ckpt_from_wandb_id(cfg.wandb_project, cfg.init_from_id)
        logger.info(f"Initializing from {init_ckpt}")
        model = BrainBertInterface.load_from_checkpoint(init_ckpt)
        if model.diff_cfg(cfg.model):
            # logger.warn("Config differs from one loaded from checkpoint. OLD config will be used")
            raise Exception('Unsupported config diff.')
        # Inject new configuration so things like new regularization params + train schedule are loaded
        # TODO not a lot of safety on the weights actually loaded
        model = BrainBertInterface.load_from_checkpoint(init_ckpt, cfg=cfg.model, strict=False)
        model.bind_io(data_attrs, cfg.model) # Bind new IO
    else:
        model = BrainBertInterface(cfg.model, data_attrs)
    if cfg.model.task.freeze_backbone:
        model.freeze_backbone()

    epochs = cfg.train.epochs
    callbacks=[
        ModelCheckpoint(
            monitor='val_loss',
            filename='val-{epoch:02d}-{val_loss:.4f}',
            save_top_k=2,
            mode='min',
            every_n_epochs=1,
            # every_n_train_steps=cfg.train.val_check_interval,
            dirpath=None
        )
    ]

    if cfg.train.patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor='val_loss',
                patience=cfg.train.patience, # Learning can be fairly slow, larger patience should allow overfitting to begin (which is when we want to stop)
                min_delta=0.00005, # we can tune this lower to squeeze a bit more..
            )
        )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    if cfg.model.lr_schedule != "fixed":
        callbacks.append(lr_monitor)

    if Metric.co_bps in cfg.model.task.metrics:
        callbacks.append(
            ModelCheckpoint(
                monitor='val_Metric.co_bps',
                filename='val_co_bps-{epoch:02d}-{val_Metric.co_bps:.4f}',
                save_top_k=2,
                mode='max',
                every_n_epochs=1,
                # every_n_train_steps=cfg.train.val_check_interval,
                dirpath=None
            )
        )

    wandb_logger = WandbLogger(project=cfg.wandb_project)

    pl.seed_everything(seed=cfg.seed)

    if cfg.train.steps:
        max_steps = cfg.train.steps
        epochs = None
    else:
        max_steps = -1

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=epochs,
        max_steps=max_steps,
        accelerator='gpu',
        devices=torch.cuda.device_count(),
        check_val_every_n_epoch=1,
        # val_check_interval=cfg.train.val_check_interval,
        callbacks=callbacks,
        default_root_dir=cfg.default_root_dir,
        track_grad_norm=2 if cfg.train.log_grad else -1,
        precision=16 if cfg.model.half_precision else 32,
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_batches,
        profiler=cfg.train.profiler if cfg.train.profiler else None,
    )

    if torch.cuda.device_count() <= 1 or trainer.global_rank == 0:
        # Note, wandb.run can also be accessed as logger.experiment but there's no benefit
        if cfg.tag:
            wandb.run.name = f'{cfg.tag}-{wandb.run.id}'
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    # === Train ===
    # num_workers = 0 # for testing
    num_workers = len(os.sched_getaffinity(0)) # If this is set too high, the dataloader may crash.
    logger.info("Preparing to fit...")
    trainer.fit(
        model,
        DataLoader(
            train, shuffle=True,
            batch_size=cfg.train.batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            collate_fn=train.collater_factory()
        ),
        DataLoader(val,
            batch_size=cfg.train.batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            collate_fn=val.collater_factory()
        ),
        ckpt_path=get_latest_ckpt_from_wandb_id(cfg.wandb_project, cfg.load_from_id) if cfg.load_from_id else None
    )
    logger.info('Run complete')

if __name__ == '__main__':
    run_exp()