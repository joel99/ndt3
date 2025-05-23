import os
import glob
import shutil

# Check for files matching the pattern './data/runs/hpc*'
# Bane of my existence.. wandb ckpts crash everything, not sure where they're coming from
hpc_files = glob.glob('./data/runs/hpc*')

# If any matching files are found, attempt to delete them
if hpc_files:
    for file in hpc_files:
        try:
            if os.path.isfile(file):
                os.remove(file)
            elif os.path.isdir(file):
                shutil.rmtree(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {str(e)}")

import sys
import copy
import signal
# Increase timeout..
os.environ["WANDB__SERVICE_WAIT"] = "300"
from typing import Dict, Any

import logging # we use top level logging since most actual diagnostic info is in libs
import hydra
from omegaconf import OmegaConf
import dataclasses

import torch
from torch.utils.data import WeightedRandomSampler
import lightning as pl
from lightning import seed_everything
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    Callback
)
from lightning.pytorch.utilities import model_summary
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning_utilities.core.rank_zero import rank_zero_only
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.plugins.environments import SLURMEnvironment

import wandb
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph() # For torch.compile, https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops

from context_general_bci.config import RootConfig, Metric, ModelTask, hp_sweep_space, propagate_config
from context_general_bci.utils import (
    lower_is_better,
    get_simple_host,
    get_scratch_path,
    generate_search,
    grid_search,
    get_best_ckpt_from_wandb_id,
    get_wandb_lineage,
    wandb_run_exists,
    wandb_get_run_exists
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
    import subprocess
    import socket
    if get_simple_host() == 'mind':
        launch_script = './slurm/launch.sh'
    elif get_simple_host() == 'nid':
        if 'falcon_h2' in cfg.experiment_set or cfg.slurm_request_str == '1x4':
            launch_script = './slurm/nersc_1x4.sh' # Quick hack
        else:
            launch_script = './slurm/nersc_basic.sh' # Hm... this hangs or gpu isn't recognized.
    else:
        if cfg.preempt:
            if 'falcon_h2' in cfg.experiment_set or cfg.slurm_request_str == '1x4':
                launch_script = './slurm/launch_1x4_preempt.sh'
            else:
                launch_script = './slurm/launch_preempt_single.sh' # These are quick jobs. Try not to burn our budget.
        else:
            if 'falcon_h2' in cfg.experiment_set or cfg.slurm_request_str == '1x4':
                launch_script = './slurm/launch_l40s.sh'
            else:
                launch_script = './slurm/launch_l40s_med.sh' # 2d jobs
                # launch_script = './slurm/launch_l40s_single.sh' # These are quick jobs. Try not to burn our budget.
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
    # breakpoint()
    if wandb_run_exists(
        cfg,
        experiment_set=flag_dict.get('experiment_set', ''),
        tag=flag_dict.get('tag', ''),
        other_overrides=config_dict,
        allowed_states=["finished", "running"]
    ):
        if cfg.cancel_if_run_exists and not cfg.debug and 'smoketest' not in cfg.tag and 'online' not in cfg.experiment_set:
            logging.info(f"Skipping {flag_dict['tag']} because it already exists.")
            return
        else:
            logging.info(f"Completed run {flag_dict['tag']} already exists, but re-running.")
    if cfg.resume_if_run_exists:
        run = wandb_get_run_exists(
            cfg,
            experiment_set=flag_dict.get('experiment_set', ''),
            tag=flag_dict.get('tag', ''),
            other_overrides=config_dict,
            allowed_states=["failed", "crashed"]
            # allowed_states=["failed", "crashed", "finished"] # graceful shutdowns from preempt are marked as finished
        )
        if run is None:
            logging.info("Requested to resume but no run found to resume")
            return
        else:
            logging.info(f"Resuming {run.name}...")
            unique_flags.append(f'load_from_id={run.name}')
    print('launching: ', ' '.join(unique_flags))
    if cfg.serial_run:
        subprocess.run(['python', 'run.py', *unique_flags])
    else:
        # print(launch_script, *unique_flags)
        env = os.environ.copy()
        subprocess.run(['sbatch', launch_script, *unique_flags], env=env)

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
    if cfg.debug:
        cfg.serial_run = True # disable slurming

    # Fragment and inherit
    # Note the order of operations. If we fragment first, we are assuming runs exist on fragmented datasets.
    # If we inherit first, we are assuming runs exist on the full dataset. try/catch full-first.
    if cfg.inherit_exp and not (cfg.load_from_id or cfg.init_from_id):
        inherit_succeeded = False
        try:
            lineage_run = get_wandb_lineage(cfg)
            if cfg.inherit_try_load:
                cfg.load_from_id = lineage_run.name
            else:
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
            cfg_trial = {'dataset.datasets': [dataset]}
            if cfg.fragment_assign_to_eval:
                cfg_trial.update({'dataset.eval_datasets': [dataset]})
            run_cfg(cfg_trial)
        exit(0)

    # Load lineage if available. Note it is essential to keep this after tag overrides above as we match with tags.
    # This is not compatible with sweeps, but should be compatible with fragment_datasets.
    # More general inherit call, above is for fragment logic
    if cfg.inherit_exp and not inherit_succeeded and not (cfg.load_from_id or cfg.init_from_id):
        lineage_run = get_wandb_lineage(cfg)
        if cfg.inherit_try_load:
            cfg.load_from_id = lineage_run.name
        else:
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
                f'experiment_set={cfg.experiment_set}',
                f'inherit_exp={cfg.inherit_exp}', # propagate identified lineage
                f'init_from_id={cfg.init_from_id}', # propagate identified lineage
                f'load_from_id={cfg.load_from_id}', # propagate identified lineage
            ]
            launcher(cfg, init_args, additional_cli_flags, meta_flags)
        if cfg.sweep_mode == 'grid':
            # Create a list of dicts from the cross product of the sweep config
            for cfg_trial in grid_search(sweep_cfg):
                run_cfg(cfg_trial)
        else:
            for cfg_trial in generate_search(sweep_cfg, cfg.sweep_trials):
                run_cfg(cfg_trial)
        exit(0)

    # Skip repeats of one-offs
    if cfg.cancel_if_run_exists and 'sweep' not in cfg.tag and not cfg.debug and 'smoketest' not in cfg.tag and wandb_run_exists(
        cfg,
        experiment_set=cfg.experiment_set,
        tag=cfg.tag,
        other_overrides={
            'config.model.lr_init': cfg.model.lr_init,
        },
        allowed_states=["finished", "running"]
    ):
        logging.info(f"Skipping this run because it already exists.")
        return

    propagate_config(cfg)
    logger = logging.getLogger(__name__)
    seed_everything(seed=cfg.seed)

    # Delay imports until necessary
    from context_general_bci.dataset import SpikingDataset, SpikingDataModule
    from context_general_bci.model import BrainBertInterface, load_from_checkpoint
    from context_general_bci.callbacks import ProbeToFineTuneEarlyStopping
    from pathlib import Path
    import os
    if cfg.slurm_use_scratch:
        load_relative_to = Path(get_scratch_path())
    else:
        load_relative_to = Path('.')
    dataset = SpikingDataset(cfg.dataset, debug=cfg.debug, load_relative_to=load_relative_to)
    dataset.build_context_index()
    if cfg.debug:
        breakpoint() # Check out the dataset
    if cfg.dataset.assert_no_crop:
        dataset.set_no_crop(True)

    if cfg.dataset.eval_datasets:
        eval_dataset = copy.deepcopy(dataset)
        eval_dataset.subset_split(splits=['eval'], keep_index=True)
    dataset.subset_split(keep_index=True)
    if cfg.debug:
        if cfg.dataset.augmentations:
            logger.info(f"Please disable augmentation to check dataset size.")
        else:
            from context_general_bci.config import DataKey
            num_timesteps = sum([dataset[i][DataKey.time].max() + 1 for i in range(len(dataset))])
            logger.info(f"Total time (s) for dataset: {int(num_timesteps * dataset.cfg.bin_size_ms // 1000)} (sans-eval)")
    dataset.subset_scale(
        limit_per_session=cfg.dataset.scale_limit_per_session,
        limit_per_eval_session=cfg.dataset.scale_limit_per_eval_session,
        limit=cfg.dataset.scale_limit,
        ratio=cfg.dataset.scale_ratio,
        keep_index=True
    )

    train, val = dataset.create_tv_datasets(train_ratio=cfg.dataset.tv_ratio)
    if cfg.dataset.replay_datasets:
        replay_cfg = copy.deepcopy(cfg.dataset)
        replay_cfg.datasets = cfg.dataset.replay_datasets
        replay_cfg.eval_datasets = []
        replay_cfg.replay_datasets = []
        replay_dataset = SpikingDataset(replay_cfg)
        replay_dataset.build_context_index()
        sampling_weight = torch.cat([
            torch.ones(len(train)) / len(train),
            torch.ones(len(replay_dataset)) * cfg.dataset.replay_weight / len(replay_dataset)
        ])
        # Note replacement _must_ be true in order to make sense of the weights
        sampler = WeightedRandomSampler(sampling_weight, replacement=True, num_samples=len(train)) # Still sample same number of trials per batch for compute / schedule equivalence
        concat = train + replay_dataset
        # Assign critical attrs
        concat.tokenized_collater = train.tokenized_collater
        concat.context_index = train.context_index

        train = concat
    else:
        sampler = None
    logger.info(f"Training on {len(train)} examples")
    data_attrs = dataset.get_data_attrs()
    # logger.info(pformat(f"Data attributes: {data_attrs}"))
    if cfg.init_from_id:
        init_ckpt = get_best_ckpt_from_wandb_id(
            cfg.wandb_project, cfg.init_from_id,
            tag=cfg.init_tag, wandb_dir=cfg.inherit_explicit_dir
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

    if cfg.train.peft_strategy == "lora":
        # https://huggingface.co/docs/peft/developer_guides/low_level_api
        from peft import inject_adapter_in_model, LoraConfig
        lora_config = LoraConfig(
            lora_alpha=cfg.train.lora_alpha,
            r=cfg.train.lora_rank,
            target_modules=cfg.train.lora_targets,
        )
        logger.info(f"Injecting LORA with config {lora_config}")
        logger.info(f"Freezing backbone...")
        # model.freeze_backbone() # Task Params are unfrozen to begin with
        model: BrainBertInterface = inject_adapter_in_model(lora_config, model)
        for name, param in model.task_pipelines.named_parameters():
            param.requires_grad = True
        logger.info('Enable the commented out Wqkv bf16 cast in `components.py` if you need to, currently disabled')

    callbacks=[]
    if cfg.save_val_loss:
        callbacks.append(
            ModelCheckpoint(
                monitor='val_loss',
                filename='val-{epoch:02d}-{val_loss:.4f}',
                save_top_k=cfg.save_num, # For rollback efforts.
                save_last=cfg.save_last,
                mode='min',
                every_n_epochs=cfg.train.val_check_epochs if cfg.train.val_save_interval == 0 else None,
                every_n_train_steps=cfg.train.val_save_interval if cfg.train.val_save_interval > 0 else None,
                dirpath=None
            ))

    if cfg.save_r2:
        if ModelTask.v_function in cfg.model.task.tasks:
            callbacks.append(
                ModelCheckpoint(
                    monitor='val_vf_loss',
                    filename='val_vf_loss-{epoch:02d}-{val_vf_loss:.4f}',
                    save_top_k=cfg.save_num,
                    save_last=cfg.save_last, # for RL
                    mode='min',
                    every_n_epochs=cfg.train.val_check_epochs if cfg.train.val_save_interval == 0 else None,
                    every_n_train_steps=cfg.train.val_save_interval if cfg.train.val_save_interval > 0 else None,
                    dirpath=None
                )
            )
        else:
            callbacks.append(
                ModelCheckpoint(
                    monitor='val_kinematic_r2',
                    filename='val_kinematic_r2-{epoch:02d}-{val_kinematic_r2:.4f}-{val_loss:.4f}',
                    save_top_k=cfg.save_num,
                    mode='max', # I'm dumb
                    save_last=cfg.save_last,
                    every_n_epochs=cfg.train.val_check_epochs if cfg.train.val_save_interval == 0 else None,
                    every_n_train_steps=cfg.train.val_save_interval if cfg.train.val_save_interval > 0 else None,
                    dirpath=None
                ),
            )
    if cfg.save_cer:
        callbacks.append(
            ModelCheckpoint(
                monitor='val_cer',
                filename='val_cer-{epoch:02d}-{val_cer:.4f}',
                save_top_k=cfg.save_num,
                save_last=cfg.save_last,
                mode='min',
                every_n_epochs=cfg.train.val_check_epochs if cfg.train.val_save_interval == 0 else None,
                every_n_train_steps=cfg.train.val_save_interval if cfg.train.val_save_interval > 0 else None,
                dirpath=None
            )
        )

    if cfg.model.lr_schedule_hotfix_epoch:
        # raise NotImplementedError("Doesn't seem to work. Max LR increases, for some reason.")
        class LRSwapCallback(Callback):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.swapped = False

            def on_train_epoch_start(self, trainer, pl_module):
                breakpoint()
                if not self.swapped and trainer.current_epoch <= cfg.model.lr_schedule_hotfix_epoch:
                    # Rather specific intervention for pytorch native lr schedulers
                    last_state_dict = pl_module.lr_schedulers().state_dict()
                    refresh_state_dict = pl_module.configure_optimizers()['lr_scheduler']['scheduler'].state_dict()
                    refresh_state_dict['_last_lr'] = [lr * cfg.model.lr_schedule_hotfix_factor for lr in refresh_state_dict['_last_lr']]
                    for i, sched in enumerate(refresh_state_dict['_schedulers']):
                        sched['_last_lr'] = [lr * cfg.model.lr_schedule_hotfix_factor for lr in last_state_dict['_schedulers'][i]['_last_lr']]
                    pl_module.lr_schedulers().load_state_dict(refresh_state_dict)
                    self.swapped = True
        callbacks.append(LRSwapCallback())

    if cfg.train.patience > 0:
        early_stop_cls = ProbeToFineTuneEarlyStopping if cfg.probe_finetune else EarlyStopping
        callbacks.append(
            early_stop_cls(
                monitor=cfg.train.early_stop_metric,
                mode='min' if lower_is_better(cfg.train.early_stop_metric) else 'max',
                patience=cfg.train.patience, # Learning can be fairly slow, larger patience should allow overfitting to begin (which is when we want to stop)
                min_delta=0,
            )
        )
        if not cfg.probe_finetune and reset_early_stop:
            def patient_load(self, state_dict: Dict[str, Any]):
                self.wait_count = 0
                # self.stopped_epoch = state_dict["stopped_epoch"]
                self.best_score = state_dict["best_score"]
                self.patience = cfg.train.patience
            import functools
            callbacks[-1].load_state_dict = functools.partial(patient_load, callbacks[-1])

    lr_monitor = LearningRateMonitor(logging_interval='step')
    if cfg.model.lr_schedule != "fixed":
        callbacks.append(lr_monitor)

    pl.seed_everything(seed=cfg.seed)

    if cfg.train.steps:
        max_steps = cfg.train.steps
        epochs = None
    else:
        max_steps = -1
        epochs = cfg.train.epochs

    if cfg.slurm_use_scratch and get_simple_host() == 'crc':
        cfg.default_root_dir = Path(get_scratch_path()) / 'data/runs'
        # Old comment was that this was "cursed";
        # But we need to revisit because we're out of non-flash storage, hah
    wandb_logger = None if cfg.debug else WandbLogger(
        project=cfg.wandb_project,
        save_dir=cfg.default_root_dir,
    )

    init_wandb(cfg, wandb_logger) # needed for checkpoint to save under wandb dir, for some reason wandb api changed.

    is_distributed = (torch.cuda.device_count() > 1) or cfg.nodes > 1
    precision = 'bf16-mixed' if cfg.model.half_precision else 32
    if is_distributed:
        from lightning.pytorch.strategies.ddp import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=len(cfg.model.task.covariate_blacklist_dims) > 0 and not cfg.dataset.tokenize_covariates)
    else:
        strategy = 'auto' # Assumes pl version 2.0+, else use None
    if cfg.train.strategy != "":
        strategy = cfg.train.strategy
    if cfg.model.full_half_precision:
        precision = 'bf16'
    # strategy = strategy="deepspeed_stage_2", precision=16
    plugins = []
    # if get_simple_host() == 'crc' and os.environ.get('SLURM_JOB_ID', None) is not None:
        # needed for sigint handling
        # plugins.append(SLURMEnvironment(auto_requeue=False)) # This thing is not actually working and disrupting my pre-empt sigterm handling for scratch moving

    # Timeouts to be picked up by nersc
    # plugins.append(SLURMEnvironment(requeue_signal=signal.SIGUSR1))

    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=epochs,
        max_steps=max_steps,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
        num_nodes=cfg.nodes,
        check_val_every_n_epoch=None if cfg.train.val_check_interval else 1,
        log_every_n_steps=cfg.train.log_every_n_steps,
        val_check_interval=cfg.train.val_check_interval if cfg.train.val_check_interval > 0 else None,
        callbacks=callbacks,
        default_root_dir=cfg.default_root_dir,
        # track_grad_norm=2 if cfg.train.log_grad else -1, # this is quite cluttered, but probably better that way. See https://github.com/Lightning-AI/lightning/issues/1462#issuecomment-1190253742 for patch if needed, though.
        precision=precision,
        strategy=strategy,
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_batches,
        profiler=cfg.train.profiler if cfg.train.profiler else None,
        overfit_batches=1 if cfg.train.overfit_batches else 0,
        plugins=plugins
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
        train, val_datasets,
        sampler=sampler
    )

    if not is_distributed and cfg.train.autoscale_batch_size: # autoscale doesn't work for DDP
        from lightning.pytorch.tuner import Tuner
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
    if cfg.train.effective_batch_size > 0 and data_module.batch_size < len(train):
        cfg.train.effective_batch_size = min(cfg.train.effective_batch_size, len(train))
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
        assert cfg.train.accumulate_batches > 0, "Accumulate batches must be greater than 0"
        logger.info(f"Accumulating {trainer.accumulate_grad_batches} batches to achieve effective batch size of {cfg.train.effective_batch_size}")
        # if cfg.model.lr_interval == 'step':
            # Not actually necessary - timm uses global step
            # logger.info('Updating LR scheduler steps to account for accumulation')
            # model.lr_schedulers() - don't think configure optimizers has even been called yet, directly override
            # cfg.model.lr_ramp_steps = cfg.model.lr_ramp_steps * cfg.train.accumulate_batches
            # cfg.model.lr_decay_steps = cfg.model.lr_decay_steps * cfg.train.accumulate_batches
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

    r"""
        Auto-requeue either resumes mid-epoch or end of epoch. Mid-epoch has stateful dataloader reqs, which doesnt' seem to be supported in stable API quite yet
        - https://github.com/pytorch/data/tree/main/torchdata/stateful_dataloader, and questionable bhvr across multinode.
        So suppose we go for end-of-epoch, which is fine because we're in the many epoch regime.
        - Auto-requeue requires 'last' to pick up most recent epoch ckpt (if it exists), otherwise will pick up automatically saved 'hpc' ckpt from SLURM requeue hook. (https://github.com/Lightning-AI/pytorch-lightning/issues/19782)
        Unsupported:
        - load_from_id is a manual resumption of a long-run. Auto-resumption will fail in this case.
        This should be fine as long as runs don't crash and resume. It implies that the auto-resumption functionality conflicts with a manual resumption.
        - For future ref, should integrate a more full-fledged SLURM system such as submitit https://github.com/facebookincubator/submitit rather than this one-off in lightning.
        - Also note that auto-resumption will override extant slurm logs, as job IDs are shared... Really not ideal.
    """
    fit_ckpt = None # use init
    if cfg.load_from_id:
        fit_ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, cfg.load_from_id, wandb_dir=cfg.inherit_explicit_dir)
    elif not cfg.init_from_id:
        fit_ckpt = None # autoresumption if possible (currently not only nonfunctional, but breaks other functions)
        # fit_ckpt = 'last' # autoresumption if possible (currently very broken)
    # print("\n\n\n\n\n\n")
    # print(f"Fit checkpoint: {fit_ckpt}")
    # print("\n\n\n\n\n\n")
    trainer.fit(
        model, datamodule=data_module,
        ckpt_path=fit_ckpt # else - autorequeue
    )
    logger.info('Run complete')
    # if cfg.slurm_use_scratch and get_simple_host() == 'crc': # Pairs with default root dir change
        # @rank_zero_only
        # def copy_back():
        #     import shutil
        #     shutil.copytree(cfg.default_root_dir / 'ndt3' / wandb.run.id, Path('.').resolve() / 'data' / 'runs' / 'ndt3' / wandb.run.id)
        #     print(f"Copying back to {Path('.').resolve() / 'data' / 'runs' / 'ndt3' / wandb.run.id}")
        # copy_back()

    if cfg.successor_exp:
        wandb.finish()
        import time
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
