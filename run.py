import os
import sys
import time
from pathlib import Path
import copy
import subprocess
import functools
import socket

from typing import Dict, Any

from pprint import pformat
import logging # we use top level logging since most actual diagnostic info is in libs
import hydra
from omegaconf import OmegaConf
import dataclasses

import torch
import lightning as pl
# import lightning.pytorch as pl
from lightning import seed_everything
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning_utilities.core.rank_zero import rank_zero_only
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.tuner import Tuner

import wandb

from context_general_bci.config import RootConfig, Metric, ModelTask, hp_sweep_space, propagate_config
from context_general_bci.dataset import SpikingDataset, SpikingDataModule
from context_general_bci.model import BrainBertInterface, load_from_checkpoint
from context_general_bci.callbacks import ProbeToFineTuneEarlyStopping
from context_general_bci.utils import (
    generate_search,
    grid_search,
    get_best_ckpt_from_wandb_id,
    get_wandb_lineage,
    wandb_run_exists
)

r"""
    For this script
    - if you're in a slurm interactive job, or want to launch a script, directly invoke
    ```
    python run.py args
    ```

    A note on usage:
    hydra will require config_path='config' and config_name='config' to load default.

    They point to using overrides to merge in experimental config.
    `python run.py +exp=test`
    where +exp directs hydra to look for "test.yaml" in ./config/exp/

    However we'd like some additional organization
    To access subdirectory experiments, the command would look like
    `python run.py +exp/subdir=test`
    (As opposed to +exp=subdir/test)
    - The latter runs but doesn't log `experiment_set` correctly
"""
reset_early_stop = True # todo move into config

@rank_zero_only
def init_wandb(cfg: RootConfig, wandb_logger):
    if cfg.debug:
        return
    # if wandb.run == None:
    #     wandb.init(project=cfg.wandb_project) # for some reason wandb changed and now I need a declaration
    _ = wandb_logger.experiment # force experiment recognition so that id is initialized

def launcher(cfg: RootConfig, init_args, additional_cli_flags, meta_flags):
    if 'mind' in socket.gethostname():
        launch_script = 'launch.sh'
    else:
        launch_script = './crc_scripts/launch_v100.sh' # assumed tiny runs if on crc?
    assembled_flags = [*init_args, *additional_cli_flags, *meta_flags]
    unique_flags = []
    seen_keys = []
    for flag in reversed(assembled_flags): # keep latest
        if "=" not in flag or flag.startswith('+'):
            unique_flags.append(flag)
        else:
            flag_name = flag.split('=')[0]
            if flag_name not in seen_keys:
                unique_flags.append(flag)
                seen_keys.append(flag_name)
    unique_flags = list(reversed(unique_flags)) # back to normal order

    # Check existence on wandb
    flag_dict = {flag.split('=')[0]: flag.split('=')[1] for flag in unique_flags if '=' in flag and not flag.startswith('+')}
    def ignore_key(k): # remove sensitive keys
        return k in ['experiment_set', 'tag', 'sweep_cfg', 'dataset.datasets', 'dataset.eval_datasets', 'inherit_exp']
    def sanitize_value(v: str):
        # Cast if possible
        try:
            return float(v)
        except:
            if v in ['True', 'False']:
                return v == 'True'
            else:
                return v

    config_dict = {f'config.{k}': sanitize_value(v) for k, v in flag_dict.items() if not ignore_key(k)}
    if cfg.cancel_if_run_exists and wandb_run_exists(
        cfg,
        experiment_set=flag_dict.get('experiment_set', ''),
        tag=flag_dict.get('tag', ''),
        other_overrides=config_dict,
        allowed_states=["finished", "running"]
    ):
        logging.info(f"Skipping {flag_dict['tag']} because it already exists.")
        return
    print('launching: ', ' '.join(unique_flags))
    if getattr(cfg, 'serial_run', False):
        subprocess.run(['python', 'run.py', *unique_flags])
    else:
        subprocess.run(['sbatch', launch_script, *unique_flags])

@hydra.main(version_base=None, config_path='context_general_bci/config', config_name="config")
def run_exp(cfg : RootConfig) -> None:
    # Check for sweeping. Note we process data above because I never intend to sweep over data config.
    if cfg.tag == "":
        r"""
            JY is used to having experimental variant names tracked with filename (instead of manually setting tags)
            take sys.argv and set the tag to the query. Only set it if we're not sweeping (where tag was already explicitly set)
        """
        exp_arg = [arg for arg in sys.argv if '+exp' in arg]
        if len(exp_arg) > 0:
            cfg.tag = exp_arg[0].split('=')[1]
        if cfg.experiment_set == "":
            cfg.experiment_set = exp_arg[0].split('=')[0][len('+exp/'):]

    # Fragment and inherit
    # Note the order of operations. If we fragment first, we are assuming runs exist on fragmented datasets.
    # If we inherit first, we are assuming runs exist on the full dataset. try/catch full-first.
    if cfg.inherit_exp:
        inherit_succeeded = False
        try:
            lineage_run = get_wandb_lineage(cfg)
            cfg.init_from_id = lineage_run.name
            cfg.inherit_exp = ""
            inherit_succeeded = True
        except:
            logging.info(f"Initial inherit for {cfg.inherit_exp} not found, pushed to post-fragment.")
    if cfg.fragment_datasets:
        def run_cfg(cfg_trial):
            init_call = sys.argv
            init_args = init_call[init_call.index('run.py')+1:]
            additional_cli_flags = [f"{k}={v}" for k, v in cfg_trial.items()] # note escaping
            meta_flags = [
                'fragment_datasets=False',
                f'tag={cfg.tag}-frag-{cfg_trial["dataset.datasets"][0]}',
                f'experiment_set={cfg.experiment_set}',
                f'inherit_exp={cfg.inherit_exp}', # propagate the following sensitive pieces
                f'init_from_id={cfg.init_from_id}' # propagate the following sensitive pieces
            ]
            if cfg.inherit_tag:
                meta_flags.append(f'inherit_tag={cfg.inherit_tag}-frag-{cfg_trial["dataset.datasets"][0]}')
            launcher(cfg, init_args, additional_cli_flags, meta_flags)

        for dataset in cfg.dataset.datasets:
            cfg_trial = {'dataset.datasets': [dataset], 'dataset.eval_datasets': [dataset]}
            run_cfg(cfg_trial)
        exit(0)

    # Load lineage if available. Note it is essential to keep this after tag overrides above as we match with tags.
    # This is not compatible with sweeps, but should be compatible with fragment_datasets.
    if cfg.inherit_exp and not inherit_succeeded:
        lineage_run = get_wandb_lineage(cfg)
        cfg.init_from_id = lineage_run.name
    if cfg.sweep_cfg: # and os.environ.get('SLURM_JOB_ID') is None: # do not allow recursive launch
        sweep_cfg = hp_sweep_space.sweep_space[cfg.sweep_cfg]
        def run_cfg(cfg_trial):
            init_call = sys.argv
            init_args = init_call[init_call.index('run.py')+1:]
            additional_cli_flags = [f'{k}={v}' for k, v in cfg_trial.items()]
            meta_flags = [
                'sweep_cfg=""',
                f'sweep_tag={cfg.sweep_cfg}',
                f'tag={cfg.tag}-sweep-{cfg.sweep_cfg}',
                f'experiment_set={cfg.experiment_set}'
            ]
            # subprocess.run(['./launch_dummy.sh', *init_args, *additional_cli_flags, *meta_flags])
            # subprocess.run(['sbatch', './crc_scripts/launch.sh', *init_args, *additional_cli_flags, *meta_flags])
            launcher(cfg, init_args, additional_cli_flags, meta_flags)
        if cfg.sweep_mode == 'grid':
            # Create a list of dicts from the cross product of the sweep config
            for cfg_trial in grid_search(sweep_cfg):
                run_cfg(cfg_trial)
        else:
            for cfg_trial in generate_search(sweep_cfg, cfg.sweep_trials):
                run_cfg(cfg_trial)
        exit(0)


    if cfg.cancel_if_run_exists and wandb_run_exists(
        cfg,
        experiment_set=cfg.experiment_set,
        tag=cfg.tag,
        other_overrides={
            'config.lr.init': cfg.model.lr_init,
            # 'config.dataset.datasets': cfg.dataset.datasets, # not needed, covered by tag
        },
        allowed_states=["finished", "running"]
    ):
        logging.info(f"Skipping this run because it already exists.")
        return

    propagate_config(cfg)
    logger = logging.getLogger(__name__)
    seed_everything(seed=cfg.seed)

    dataset = SpikingDataset(cfg.dataset)
    dataset.build_context_index()
    if cfg.dataset.eval_datasets:
        eval_dataset = copy.deepcopy(dataset)
        eval_dataset.subset_split(splits=['eval'], keep_index=True)
    dataset.subset_split(keep_index=True)
    if cfg.dataset.scale_limit_per_session or cfg.dataset.scale_limit_per_eval_session:
        dataset.subset_scale(
            limit_per_session=cfg.dataset.scale_limit_per_session,
            limit_per_eval_session=cfg.dataset.scale_limit_per_eval_session,
            keep_index=True
        )
    elif cfg.dataset.scale_limit:
        dataset.subset_scale(limit=cfg.dataset.scale_limit, keep_index=True)
    elif cfg.dataset.scale_ratio:
        dataset.subset_scale(ratio=cfg.dataset.scale_ratio, keep_index=True)
    train, val = dataset.create_tv_datasets()
    logger.info(f"Training on {len(train)} examples")
    data_attrs = dataset.get_data_attrs()
    # logger.info(pformat(f"Data attributes: {data_attrs}"))

    if cfg.init_from_id:
        init_ckpt = get_best_ckpt_from_wandb_id(
            cfg.wandb_project, cfg.init_from_id,
            tag=cfg.init_tag
        )
        logger.info(f"Initializing from {init_ckpt}")
        model = load_from_checkpoint(init_ckpt, cfg=cfg.model, data_attrs=data_attrs)
    elif cfg.init_ckpt:
        logger.info(f"Initializing from {cfg.init_ckpt}")
        model = load_from_checkpoint(cfg.init_ckpt, cfg=cfg.model, data_attrs=data_attrs)
    else:
        model = BrainBertInterface(cfg.model, data_attrs)
    if cfg.model.task.freeze_embed:
        model.freeze_embed()
    if cfg.model.task.freeze_backbone:
        model.freeze_backbone()
    if cfg.model.task.freeze_all:
        model.freeze_non_embed()

    callbacks=[
        ModelCheckpoint(
            monitor='val_loss',
            filename='val-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            mode='min',
            every_n_epochs=1,
            # every_n_train_steps=cfg.train.val_check_interval,
            dirpath=None
        ),
    ]
    # Eh, this doesn't produce different results.
    # if ModelTask.kinematic_decoding in cfg.model.task.tasks:
    #     callbacks.append(
    #         ModelCheckpoint(
    #         monitor='val_kinematic_decoding_loss',
    #             filename='val_kin-{epoch:02d}-{val_loss:.4f}',
    #             save_top_k=1,
    #             mode='min',
    #             every_n_epochs=1,
    #             # every_n_train_steps=cfg.train.val_check_interval,
    #             dirpath=None
    #         ),
    #     )

    if cfg.train.patience > 0:
        early_stop_cls = ProbeToFineTuneEarlyStopping if cfg.probe_finetune else EarlyStopping
        callbacks.append(
            early_stop_cls(
                monitor='val_loss',
                patience=cfg.train.patience, # Learning can be fairly slow, larger patience should allow overfitting to begin (which is when we want to stop)
                min_delta=1e-5,
            )
        )
        if not cfg.probe_finetune and reset_early_stop:
            def patient_load(self, state_dict: Dict[str, Any]):
                self.wait_count = 0
                # self.stopped_epoch = state_dict["stopped_epoch"]
                self.best_score = state_dict["best_score"]
                self.patience = cfg.train.patience
            callbacks[-1].load_state_dict = functools.partial(patient_load, callbacks[-1])

    lr_monitor = LearningRateMonitor(logging_interval='step')
    if cfg.model.lr_schedule != "fixed":
        callbacks.append(lr_monitor)

    for m in [Metric.co_bps, Metric.bps]:
        if m in cfg.model.task.metrics:
            callbacks.append(
                ModelCheckpoint(
                    monitor=f'val_{m.value}',
                    filename='val_' + m.value + '-{epoch:02d}-{val_' + m.value + ':.4f}',
                    save_top_k=1,
                    mode='max',
                    every_n_epochs=1,
                    # every_n_train_steps=cfg.train.val_check_interval,
                    dirpath=None
                )
            )

    pl.seed_everything(seed=cfg.seed)

    if cfg.train.steps:
        max_steps = cfg.train.steps
        epochs = None
    else:
        max_steps = -1
        epochs = cfg.train.epochs

    wandb_logger = None if cfg.debug else WandbLogger(
        project=cfg.wandb_project,
        save_dir=cfg.default_root_dir,
    )

    init_wandb(cfg, wandb_logger) # needed for checkpoint to save under wandb dir, for some reason wandb api changed.

    is_distributed = (torch.cuda.device_count() > 1) or getattr(cfg, 'nodes', 1) > 1
    default_strat = 'auto' if pl.__version__.startswith('2.0') else None
    precision = 'bf16-mixed' if cfg.model.half_precision else 32
    strategy = DDPStrategy(find_unused_parameters=len(cfg.model.task.covariate_blacklist_dims) > 0 and not cfg.dataset.tokenize_covariates) if is_distributed else default_strat
    if cfg.train.strategy != "":
        strategy = cfg.train.strategy
    # if cfg.model.full_half_precision:
        # precision = 'bf16'
    # strategy = strategy="deepspeed_stage_2", precision=16
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=epochs,
        max_steps=max_steps,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
        num_nodes=getattr(cfg, 'nodes', 1),
        check_val_every_n_epoch=1,
        log_every_n_steps=cfg.train.log_every_n_steps,
        # val_check_interval=cfg.train.val_check_interval,
        callbacks=callbacks,
        default_root_dir=cfg.default_root_dir,
        # track_grad_norm=2 if cfg.train.log_grad else -1, # this is quite cluttered, but probably better that way. See https://github.com/Lightning-AI/lightning/issues/1462#issuecomment-1190253742 for patch if needed, though.
        precision=precision,
        strategy=strategy,
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_batches,
        profiler=cfg.train.profiler if cfg.train.profiler else None,
        overfit_batches=1 if cfg.train.overfit_batches else 0
    )

    # assumes Tensor cores
    torch.set_float32_matmul_precision('medium') # we don't care about precision really..

    # === Train ===
    num_workers = min(len(os.sched_getaffinity(0)), 16) # If this is set too high, the dataloader may crash.
    if cfg.debug:
        logger.warning("\n \n \n Num workers is 0, DEBUGGING.")
        num_workers = 0 # for testing
        logger.warning("Num workers is 0, DEBUGGING. \n \n \n")
    logger.info("Preparing to fit...")

    val_datasets = [val]
    if cfg.dataset.eval_datasets:
        val_datasets.append(eval_dataset)

    if cfg.train.effective_batch_size > 0:
        if not is_distributed:
            if not cfg.train.autoscale_batch_size:
                logger.warning("You should probably take advantage of autoscale_batch_size if setting effective batch size")
            else:
                cfg.train.batch_size = 1

    data_module = SpikingDataModule(
        cfg.train.batch_size,
        num_workers,
        train, val_datasets
    )

    if not is_distributed and cfg.train.autoscale_batch_size: # autoscale doesn't work for DDP
        tuner = Tuner(trainer)
        tuner.scale_batch_size(
            model,
            datamodule=data_module,
            mode="power",
            init_val=2, # Mostly convenient for iteration. Eventually we'll be stuck at bsz 1.
            steps_per_trial=15,
        )
        if cfg.train.max_batch_size and data_module.batch_size > cfg.train.max_batch_size:
            data_module.batch_size = min(data_module.batch_size, cfg.train.max_batch_size)
            print(f'Clip down max batch size to  {data_module.batch_size}')

    # Compute necessary accumulation, if prescribed.
    if cfg.train.effective_batch_size > 0:
        assert cfg.train.accumulate_batches == 1, "Cannot specify both effective_batch_size and accumulate_batches"
        if cfg.train.effective_batch_size < cfg.train.batch_size:
            raise ValueError(f"Effective batch size {cfg.train.effective_batch_size} must be larger than (probably autoscaled) batch size {cfg.train.batch_size}")
        if is_distributed:
            replicas = cfg.nodes * torch.cuda.device_count()
            logger.info(f"Running on {replicas} replicas")
            if data_module.batch_size * replicas > cfg.train.effective_batch_size:
                raise ValueError(f"Effective batch size {cfg.train.effective_batch_size} must be larger than (probably autoscaled) batch size {data_module.batch_size * replicas}")
            cfg.train.accumulate_batches = int(cfg.train.effective_batch_size / (data_module.batch_size * replicas))
        else:
            cfg.train.batch_size = data_module.batch_size
            # Autotune, then determine
            cfg.train.accumulate_batches = int(cfg.train.effective_batch_size / cfg.train.batch_size)
        trainer.accumulate_grad_batches = cfg.train.accumulate_batches
        logger.info(f"Accumulating {trainer.accumulate_grad_batches} batches to achieve effective batch size of {cfg.train.effective_batch_size}")
        # ! note this post-hoc update... reliant on the Trainer api using this prop
        # https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/callbacks/gradient_accumulation_scheduler.html#GradientAccumulationScheduler

    # Log parameters for reference
    cfg.trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg.total_parameters = sum(p.numel() for p in model.parameters())

    # Note, wandb.run can also be accessed as logger.experiment but there's no benefit
    # torch.cuda.device_count() > 1 or cfg.nodes > 1
    if trainer.global_rank == 0 and not cfg.debug:
        logger.info(f"Running NDT2, dumping config:")
        logger.info(OmegaConf.to_yaml(cfg))
        if cfg.tag:
            wandb.run.name = f'{cfg.tag}-{wandb.run.id}'
        notes = cfg.notes
        if os.environ.get('SLURM_JOB_ID'):
            wandb.run.notes = f"{notes}. SLURM_JOB_ID={os.environ['SLURM_JOB_ID']}"
            cfg.slurm_id = int(os.environ['SLURM_JOB_ID'])
            if not cfg.train.autoscale_batch_size:
                cfg.train.effective_batch_size = cfg.train.batch_size * cfg.train.accumulate_batches * cfg.nodes * torch.cuda.device_count()
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        wandb.config.update({'data_attrs': dataclasses.asdict(data_attrs)})
        # Of course now I find these
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("eval_loss", summary="min")
        wandb.define_metric(f"val_{Metric.bps.value}", summary="max")
        wandb.define_metric(f"eval_{Metric.bps.value}", summary="max")
        wandb.define_metric(f"val_{Metric.kinematic_r2.value}", summary="max")
        wandb.define_metric(f"eval_{Metric.kinematic_r2.value}", summary="max")

    if cfg.model.compile:
        model = torch.compile(model)
    trainer.fit(
        model, datamodule=data_module,
        ckpt_path=get_best_ckpt_from_wandb_id(cfg.wandb_project, cfg.load_from_id) if cfg.load_from_id else None
    )
    logger.info('Run complete')

    if cfg.successor_exp:
        wandb.finish()
        time.sleep(30)
        logger.info(f"Running successor experiment {cfg.successor_exp[0]}")
        # Find current filename, and locate in successor dir
        init_call = sys.argv
        init_args = init_call[init_call.index('run.py')+1:]
        # wipe sensitive CLI args here
        def should_refresh(x: str):
            return x.startswith('+exp/') or x.startswith('inherit_exp') or x.startswith('init_from_id')
            # successors will infer the fresh exp
        init_args = [x for x in init_args if not should_refresh(x)]
        tag_root = cfg.tag
        if 'frag' in tag_root:
            tag_root = tag_root[:tag_root.index('frag') - 1]
        exp_arg = f'+exp/{cfg.successor_exp[0]}={tag_root}'
        meta_flags = [
            f"experiment_set={cfg.successor_exp[0]}",
            f'successor_exp={cfg.successor_exp[1:]}',
        ]
        launcher(cfg, [exp_arg], init_args, meta_flags)


if __name__ == '__main__':
    run_exp()
