#%%
# Autoregressive inference procedure, for generalist model
import os
import argparse
from pprint import pprint

import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
import seaborn as sns
import torch
torch.set_warn_always(False) # Turn off warnings, we get cast spam otherwise
from sklearn.metrics import r2_score

from torch.utils.data import DataLoader
import lightning.pytorch as pl

from context_general_bci.model import transfer_model
from context_general_bci.dataset import SpikingDataset
from context_general_bci.config import RootConfig, ModelConfig, ModelTask, Metric, Output, DataKey, MetaKey
from context_general_bci.contexts import context_registry

from context_general_bci.analyze_utils import stack_batch, load_wandb_run, prep_plt
from context_general_bci.utils import get_wandb_run, wandb_query_latest

def main(
    student: bool,
    temperature: float,
    id: int,
    gpu: int,
    cue: float,
    batch_size: int,
):
    print("Starting eval")
    print(f"Student: {student}")
    print(f"Temperature: {temperature}")
    print(f"ID: {id}")
    print(f"GPU: {gpu}")
    print(f"Cue: {cue}")

    wandb_run = wandb_query_latest(id, allow_running=True, use_display=True)[0]
    print(wandb_run.id)

    src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_loss')

    # cfg.model.task.metrics = [Metric.kinematic_r2]
    cfg.model.task.outputs = [Output.behavior, Output.behavior_pred]

    target = [
        # 'miller_Jango-Jango_20150730_001',

        # 'dyer_co_.*',
        # 'dyer_co_mihi_1',
        # 'gallego_co_Chewie_CO_20160510',
        'gallego_co_.*',
        # 'churchland_misc_jenkins-10cXhCDnfDlcwVJc_elZwjQLLsb_d7xYI',
        # 'churchland_maze_nitschke-sub-Nitschke_ses-20090812',
        # 'churchland_maze_jenkins.*',
        # 'odoherty_rtt-Indy-20160407_02',
        # 'odoherty_rtt-Indy-20161026_03',
        # 'odoherty_rtt-Loco-20170215_02',
        # 'odoherty_rtt-Loco-20170216_02',
        # 'odoherty_rtt-Loco-20170217_02',
        # 'pitt_broad_pitt_co_CRS02bLab_1899', # Some error here. But this is 2d, so leaving for now...
        # 'pitt_broad_pitt_co_CRS02bLab_1761',
        # 'pitt_broad_pitt_co_CRS07Home_32',
        # 'pitt_broad_pitt_co_CRS07Home_88',
        # 'pitt_broad_pitt_co_CRS02bLab_1776_1.*'
    ]

    # Note: This won't preserve train val split, try to make sure eval datasets were held out
    cfg.dataset.datasets = target
    dataset = SpikingDataset(cfg.dataset)
    pl.seed_everything(0)
    # Quick cheese - IDR how to subset by length, so use "val" to get 20% quickly
    dataset.subset_scale(limit_per_session=96)
    # train, val = dataset.create_tv_datasets()
    # dataset = val
    print("Eval length: ", len(dataset))
    data_attrs = dataset.get_data_attrs()
    print(data_attrs)

    model = transfer_model(src_model, cfg.model, data_attrs)

    model.cfg.eval.teacher_timesteps = int(50 * cue) # 0.5s
    # model.cfg.eval.teacher_timesteps = int(50 * 0.1) # 0.5s
    # model.cfg.eval.teacher_timesteps = int(50 * 0.) # 0.5s
    # model.cfg.eval.teacher_timesteps = int(50 * 2) # 2s
    model.cfg.eval.limit_timesteps = 50 * 4 # up to 4s
    # model.cfg.eval.limit_timesteps = 50 * 5 # up to 4s
    model.cfg.eval.temperature = temperature
    model.cfg.eval.use_student = student

    trainer = pl.Trainer(accelerator='gpu', devices=[gpu], default_root_dir='./data/tmp')
    def get_dataloader(dataset: SpikingDataset, batch_size=batch_size, num_workers=1, **kwargs) -> DataLoader:
        return DataLoader(dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            collate_fn=dataset.tokenized_collater,
        )

    dataloader = get_dataloader(dataset)
    heldin_outputs = stack_batch(trainer.predict(model, dataloader))
    # print(heldin_outputs[Output.behavior_pred].shape)
    # print(heldin_outputs[Output.behavior].shape)

    prediction = heldin_outputs[Output.behavior_pred]
    target = heldin_outputs[Output.behavior]
    is_student = heldin_outputs[Output.behavior_query_mask]
    # Compute R2
    r2 = r2_score(target, prediction)
    r2_student = r2_score(target[is_student], prediction[is_student])
    print(f'R2: {r2:.4f}')
    print(f'R2 Student: {r2_student:.4f}')
    pprint(model.cfg.eval)
    # print(query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script.")

    parser.add_argument("-s", "--student", action="store_true", help="Flag indicating if the subject is a student.")
    parser.add_argument("-t", "--temperature", type=float, default=0., help="Temperature value.")
    parser.add_argument("-i", "--id", type=str, required=True, help="ID number.")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU index.")
    parser.add_argument("-c", "--cue", type=float, default=0.5, help="Cue context length (s)" )
    parser.add_argument("-b", "--batch_size", type=int, default=48, help="Batch size.")

    args = parser.parse_args()
    main(**vars(args))

"""
query = 'monkey_trialized_6l_1024-zgsjsog0'

# query = 'monkey_trialized_6l_1024_broad-3x3mrjdh'
# query = 'monkey_trialized_6l_1024_broad-yy3ve3gf'
# query = 'monkey_trialized_6l_1024_all-ufyxs032'

query = 'monkey_nomask_6l_1024-zfwshzmr'
# query = 'monkey_schedule_6l_1024-0swiit7z'
# query = 'monkey_kin_6l_1024-vgdhzzxm'
# query = 'monkey_random_6l_1024-n3f68hj2'
# query = 'monkey_schedule_6l_1024-7o3bb4z8'
query = 'monkey_tune_6l_1024-wy62dj4v'
"""