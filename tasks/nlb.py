from typing import Dict, List, Any
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import torch
from einops import rearrange

import logging
logger = logging.getLogger(__name__)

try:
    from nlb_tools.nwb_interface import NWBDataset
    from nlb_tools.make_tensors import (
        make_train_input_tensors,
        make_eval_input_tensors,
        make_eval_target_tensors,
        PARAMS,
        _prep_mask,
        make_stacked_array
    )
except:
    logger.info("nlb_tools not installed, please install with `pip install nlb-tools`")
    NWBDataset = None

from ..config import DataKey, DatasetConfig, MetaKey
from ..subjects import SubjectInfo, SubjectName, SubjectArrayRegistry, create_spike_payload
from ..tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry

TrialNum = int
MetadataKey = str




# Core loading strategy pulled from https://github.com/neurallatents/nlb_tools/blob/main/examples/tutorials/basic_example.ipynb

class NLBLoader(ExperimentalTaskLoader):
    name = "nlb_base"

    @classmethod
    def load(
        cls,
        datapath: Path,
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        context_arrays: List[str],
        dataset_alias: str,
        task: ExperimentalTask,
        phase='test',
        dataset_cls=NWBDataset,
        make_tensor_fn=make_train_input_tensors,
        make_test_tensor_fn=make_eval_input_tensors,
    ):
        r"""
            Loader for motor tasks in Neural Latents Benchmark (NLB) dataset.
        """
        dataset = dataset_cls(datapath)
        dataset.resample(cfg.bin_size_ms)

        # Create suffix for group naming later
        suffix = '' if (cfg.bin_size_ms == 5) else f'_{int(cfg.bin_size_ms)}'
        train_split = 'train' if (phase == 'val') else ['train', 'val']
        train_dict = make_tensor_fn(
            dataset,
            dataset_name=dataset_alias,
            trial_split=train_split,
            save_file=False
        )

        test_dict = make_test_tensor_fn(
            dataset,
            dataset_name=dataset_alias,
            trial_split='test',
            save_file=False,
        )

        if DataKey.bhvr_vel in cfg.data_keys:
            train_bhvr = make_eval_target_tensors(
                dataset,
                dataset_name=dataset_alias,
                train_trial_split=train_split,
                eval_trial_split=['val'], # we ignore this in return; function breaks with no split
                update_params={
                    'lag': 0 # we do our own lag processing
                },
                save_file=False,
                include_psth=False,
            )[f'{dataset_alias}{suffix}']['train_behavior'] # B T H=2
            train_bhvr = train_bhvr / -1000 # # felix prepared these to be +mm/s, convert to -m/s for consistency with my preparation
            train_bhvr = torch.tensor(train_bhvr, dtype=torch.float32)

        # Show fields of returned dict
        # print(train_dict.keys())

        # Unpack data
        train_spikes_heldin = train_dict['train_spikes_heldin']
        train_spikes_heldout = train_dict['train_spikes_heldout']
        test_spikes_heldin = test_dict['eval_spikes_heldin']
        # Print 3d array shape: trials x time x channel
        # print(train_spikes_heldin.shape)
        train_spikes_heldin = torch.tensor(train_spikes_heldin, dtype=torch.uint8)
        train_spikes_heldout = torch.tensor(train_spikes_heldout, dtype=torch.uint8)
        test_spikes_heldin = torch.tensor(test_spikes_heldin, dtype=torch.uint8)
        meta_payload = {}
        meta_payload['path'] = []
        meta_payload['split'] = []

        arrays_to_use = context_arrays

        for trial in range(train_spikes_heldin.shape[0]):
            spikes = rearrange(train_spikes_heldin[trial], 't c -> t c 1')
            heldout_spikes = rearrange(train_spikes_heldout[trial], 't c -> t c 1')
            single_payload = {
                DataKey.spikes: create_spike_payload(spikes, arrays_to_use),
                DataKey.heldout_spikes: heldout_spikes.clone()
            }
            if DataKey.bhvr_vel in cfg.data_keys:
                single_payload[DataKey.bhvr_vel] = train_bhvr[trial].clone()
            single_path = cache_root / f"{trial}.pth"
            meta_payload['path'].append(single_path)
            meta_payload['split'].append('train')
            torch.save(single_payload, single_path)
        trial_count = train_spikes_heldin.shape[0]
        for trial in range(test_spikes_heldin.shape[0]):
            spikes = rearrange(test_spikes_heldin[trial], 't c -> t c 1')
            single_payload = {
                DataKey.spikes: create_spike_payload(spikes, arrays_to_use),
            }
            single_path = cache_root / f"test_{trial_count + trial}.pth"
            meta_payload['path'].append(single_path)
            meta_payload['split'].append('test')
            torch.save(single_payload, single_path)

        return pd.DataFrame(meta_payload)

@ExperimentalTaskRegistry.register
class MazeLoader(NLBLoader):
    name = ExperimentalTask.nlb_maze

@ExperimentalTaskRegistry.register
class RTTLoader(NLBLoader):
    name = ExperimentalTask.nlb_rtt
