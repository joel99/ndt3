from typing import Dict, Any
import torch
from torch.optim.optimizer import Optimizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    BaseFinetuning,
)

class ProbeToFineTuneFinetuning(BaseFinetuning):
    def __init__(self):
        super().__init__()
        self._should_unfreeze = False

    @property
    def should_unfreeze(self):
        return self._should_unfreeze

    def unfreeze(self):
        self._should_unfreeze = True

    @staticmethod
    def _add_if_exists(pl_module, attrs):
        out = []
        for attr in attrs:
            if hasattr(pl_module, attr):
                out.append(getattr(pl_module, attr))
        return out

    def finetune_function(self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer, opt_idx: int) -> None:
        if self.should_unfreeze:
            for attr in ['session_flag', 'subject_flag', 'task_flag']:
                if hasattr(pl_module, attr):
                    getattr(pl_module, attr).requires_grad = True # more for the principle
            tasks_to_unfreeze = [pl_module.task_pipelines[k] for k in pl_module.task_pipelines if 'infill' in str(k)]
            # import pdb;pdb.set_trace()
            self.unfreeze_and_add_param_group(
                modules=[
                    *self._add_if_exists(pl_module, [
                        'backbone', 'readin'
                    ]),
                    *tasks_to_unfreeze,
                ],
                optimizer=optimizer,
                initial_denom_lr=1,
                train_bn=True,
            )
            self._should_unfreeze = False
            pl_module.detach_backbone_for_task = False
            # pl_module.cfg.task.task_weights[0] = 0. # kill unsup loss
            # import pdb;pdb.set_trace()
            # Actually, nuke the optimizer state (as though fresh run)
            # ! We scrap this is just wipe the optimizer in trainer class

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        tasks_to_freeze = [pl_module.task_pipelines[k] for k in pl_module.task_pipelines if 'infill' in str(k)]
        for attr in ['session_flag', 'subject_flag', 'task_flag']:
            if hasattr(pl_module, attr):
                getattr(pl_module, attr).requires_grad = False
        self.freeze([
            *self._add_if_exists(pl_module, [
                'backbone', 'readin'
            ]),
            *tasks_to_freeze,
        ])
        pl_module.detach_backbone_for_task = True

        # the unfrozen should be just context embed, target task (typically just an `out` layer)
class ProbeToFineTuneEarlyStopping(EarlyStopping):
    r"""
        Support two stage fine-tuning.
        First, fit any probe parameters.
        - when converged, rollback to best val checkpoint, then switch to stage 2.
        Second, fit all parameters.

        Implemented by merging functionality of EarlyStopping and BaseFinetuning.
        Assumes constant LR, i.e. doesn't mess with state of other callbacks.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finetune_callback = ProbeToFineTuneFinetuning()
        self.monitor = 'val_loss'
        self.stage = 1

    def state_dict(self):
        return {
            **super().state_dict(),
            **self.finetune_callback.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.finetune_callback.load_state_dict(state_dict)

    def on_fit_start(self, trainer, pl_module):
        super().on_fit_start(trainer, pl_module)
        self.finetune_callback.on_fit_start(trainer, pl_module)

    def setup(self, trainer, pl_module, stage: str):
        super().setup(trainer, pl_module, stage)
        self.finetune_callback.setup(trainer, pl_module, stage)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_start(trainer, pl_module)
        self.finetune_callback.on_train_epoch_start(trainer, pl_module)

    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        super()._run_early_stopping_check(trainer)
        if trainer.should_stop:
            if self.stage == 1:
                print("Probe fit, rolling back to best val checkpoint for finetuning...")
                # Iterate through callbacks, looking for val checkpoint callback
                # Pretty illegal...
                rollback_path = ""
                for callback in trainer.callbacks:
                    if isinstance(callback, ModelCheckpoint):
                        if callback.monitor == self.monitor:
                            rollback_path = callback.best_model_path
                if rollback_path == "":
                    raise ValueError("Could not find ModelCheckpoint callback with monitor=val_loss")
                # scraped from BatchSizeFinder source code
                # epoch = trainer.current_epoch
                trainer._checkpoint_connector.restore(rollback_path)
                # trainer.current_epoch = epoch
                trainer.should_stop = False
                self.stage = 2
                self.wait_count = 0
                self.patience = 10 * self.patience
                self.finetune_callback.unfreeze() # unfreeze all of model
                # wipe optimizer state
                trainer.optimizers = [torch.optim.AdamW(trainer.model.parameters(), lr=trainer.model.cfg.lr_init)]
                # trainer.optimizers = [torch.optim.AdamW(trainer.model.parameters(), lr=trainer.model.cfg.lr_init * 0.1)]
                # import pdb;pdb.set_trace()