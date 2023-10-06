r"""
    Miller/Limb lab data under XDS format e.g.
    https://datadryad.org/stash/dataset/doi:10.5061/dryad.cvdncjt7n (Jango, force isometric, 20 sessions, 95 days)
    Data proc libs:
    - https://github.com/limblab/xds
    - https://github.com/limblab/adversarial_BCI/blob/main/xds_tutorial.ipynb
    JY updated the xds repo into a package, clone here: https://github.com/joel99/xds/

    Features EMG data and abundance of isometric tasks.
    No fine-grained analysis - just cropped data for pretraining.
"""
from typing import List
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch
try:
    import xds_python as xds
except:
    logging.info("XDS not installed, please install from https://github.com/joel99/xds/. Import will fail")
    xds = None

from einops import reduce, rearrange

from context_general_bci.config import DataKey, DatasetConfig, REACH_DEFAULT_KIN_LABELS, EMG_CANON_LABELS
from context_general_bci.subjects import SubjectInfo, SubjectArrayRegistry, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry

MILLER_LABELS = [*REACH_DEFAULT_KIN_LABELS, *EMG_CANON_LABELS]
# No sanitation needed implemented at the moment, only using curated data
# TODO add guards
@ExperimentalTaskRegistry.register
class MillerLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.miller

    @classmethod
    def load(
        cls,
        datapath: Path,
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        context_arrays: List[str],
        dataset_alias: str,
        **kwargs,
    ):
        my_xds = xds.lab_data(str(datapath.parent), str(datapath.name)) # Load the data using the lab_data class in xds.py
        assert cfg.bin_size_ms % (my_xds.bin_width * 1000) == 0, "bin_size_ms must divide bin_size in the data"
        # We do resizing using xds native utilities, not our chopping mechanisms
        my_xds.update_bin_data(cfg.bin_size_ms / 1000) # rebin to 20ms

        meta_payload = {}
        meta_payload['path'] = []
        global_args = {}

        if cfg.tokenize_covariates:
            canonical_labels = []
            if my_xds.has_cursor:
                canonical_labels.extend(REACH_DEFAULT_KIN_LABELS)
            if my_xds.has_EMG:
                # Use muscle labels
                canonical_labels.extend(my_xds.EMG_names)
                for i, label in enumerate(my_xds.EMG_names):
                    assert label in EMG_CANON_LABELS, f"EMG label {label} not in canonical labels, please regiser in `config_base` for bookkeeping."
            if my_xds.has_force:
                # Cursor, EMG (we don't include manipulandum force, mostly to stay under 10 dims for now)
                logger.info('Force data found but not loaded for now')
            global_args[DataKey.covariate_labels] = canonical_labels

        # Print total active time etc
        all_trials = [*my_xds.get_trial_info('R'), *my_xds.get_trial_info('F')] # 'A' not included
        end_times = [trial['trial_end_time'] for trial in all_trials]
        start_times = [trial['trial_gocue_time'] for trial in all_trials]
        # ? Does the end time indicate the sort of... bin count?
        total_time = sum([end - start for start, end in zip(start_times, end_times)])
        print(f'Total trial/active time: {total_time:.2f} / {(my_xds.time_frame[-1] - my_xds.time_frame[0])[0]:.2f}')
        def chop_vector(vec: torch.Tensor):
            # vec - already at target resolution, just needs chopping
            chops = round(cfg.miller.chop_size_ms / cfg.bin_size_ms)
            return rearrange(
                vec.unfold(0, chops, chops),
                'trial hidden time -> trial time hidden'
             ) # Trial x C x chop_size (time)
        vel_pieces = []
        if my_xds.has_cursor:
            vel_pieces.append(torch.as_tensor(my_xds.curs_v, dtype=torch.float))
        if my_xds.has_EMG:
            vel_pieces.append(torch.as_tensor(my_xds.EMG, dtype=torch.float))
        vel = torch.cat(vel_pieces, 1) # T x H
        if cfg.miller.minmax:
            # Aggregate velocities and get min/max
            global_args['cov_mean'] = vel.mean(0)
            global_args['cov_min'] = torch.quantile(vel, 0.001, dim=0) # sufficient in a quick notebook check.
            global_args['cov_max'] = torch.quantile(vel, 0.999, dim=0)
            rescale = global_args['cov_max'] - global_args['cov_min']
            rescale[torch.isclose(rescale, torch.tensor(0.))] = 1
            vel = (vel - global_args['cov_mean']) / rescale
            vel = torch.clamp(vel, -1, 1)

        vel = chop_vector(vel) # T x H
        full_spikes = chop_vector(torch.as_tensor(my_xds.spike_counts, dtype=torch.float))
        # breakpoint()
        for t in range(full_spikes.size(0)):
            single_payload = {
                DataKey.spikes: create_spike_payload(full_spikes[t], context_arrays),
                DataKey.bhvr_vel: vel[t].clone(), # T x H
                **global_args,
            }
            single_path = cache_root / f'{t}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)