#%%
from typing import List
from pathlib import Path
import math
import numpy as np
import torch
import pandas as pd
from einops import rearrange, reduce

import logging
logger = logging.getLogger(__name__)
try:
    from pynwb import NWBHDF5IO
except:
    logger.info("pynwb not installed, please install with `conda install -c conda-forge pynwb`")

from context_general_bci.config import DataKey, DatasetConfig
from context_general_bci.subjects import SubjectInfo, SubjectArrayRegistry, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry



BLACKLIST_UNITS = [1]
@ExperimentalTaskRegistry.register
class ChurchlandMazeLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.churchland_maze
    r"""
    Churchland/Kaufman reaching data.
    # https://dandiarchive.org/dandiset/000070/draft/files

    Initial exploration done in `churchland_debug.py`.
    We write a slightly different loader rather than use NLB loader
    for a bit more granular control.
    """

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
        sampling_rate: int = 1000 # Hz
    ):
        if cfg.churchland_maze.chop_size_ms > 0:
            assert cfg.churchland_maze.chop_size_ms % cfg.bin_size_ms == 0, "Chop size must be a multiple of bin size"
            # if 0, no chop, just send in full lengths
        assert cfg.churchland_maze.load_covariates == False, "Covariates not supported yet"

        with NWBHDF5IO(datapath, 'r') as io:
            nwbfile = io.read()
            trial_info = nwbfile.trials
            is_valid = ~(trial_info['discard_trial'][:].astype(bool))
            move_begins = trial_info['move_begins_time'][:]
            move_ends = trial_info['move_ends_time'][:]
            trial_ends = trial_info['stop_time'][:]
            end_time_mapped = np.isnan(move_ends)
            move_ends[end_time_mapped] = trial_ends[end_time_mapped]
            spikes = nwbfile.units.to_dataframe()
            spike_intervals = spikes.obs_intervals # 1 per unit
            spike_times = spikes.spike_times # 1 per unit

            move_begins = move_begins[is_valid]
            move_ends = move_ends[is_valid]
            for t in range(len(spike_intervals)):
                spike_intervals.iloc[t] = spike_intervals.iloc[t][is_valid]

        meta_payload = {}
        meta_payload['path'] = []

        drop_units = BLACKLIST_UNITS
        # Validation
        def is_ascending(times):
            return np.all(np.diff(times) > 0)
        reset_time = 0
        if not is_ascending(move_begins):
            first_nonascend = np.where(np.diff(move_begins) <= 0)[0][0]
            logger.warning(f"Move begins not ascending, cropping to ascending {(100 * first_nonascend / len(move_begins)):.2f} %")
            move_begins = move_begins[:first_nonascend]
            move_ends = move_ends[:first_nonascend]
            is_valid = is_valid[:first_nonascend]
            reset_time = move_begins[-1]
            # No need to crop obs_intervals, we'll naturally only index so far in
        for mua_unit in range(len(spike_times)):
            if not is_ascending(spike_times.iloc[mua_unit]) and mua_unit not in drop_units:
                reset_idx = (spike_times.iloc[mua_unit] > reset_time).nonzero()
                if len(reset_idx) > 0:
                    reset_idx = reset_idx[0][0]
                    logger.warning(f"Spike times for unit {mua_unit} not ascending, crop to {(100 * reset_idx / len(spike_times.iloc[mua_unit])):.2f}% of spikes")
                    spike_times.iloc[mua_unit] = spike_times.iloc[mua_unit][:reset_idx]
                else:
                    logger.warning(f"No reset index found for unit {mua_unit}! Skipping...")
                    drop_units.append(mua_unit)
                # Based on explorations, several of the datasets have repeated trials / unit times. All appear to complete/get to a fairly high point before resetting


        for t in range(len(spike_intervals)):
            assert (spike_intervals.iloc[t] == spike_intervals.iloc[0]).all(), "Spike intervals not equal"
        spike_intervals = spike_intervals.iloc[0] # all equal in MUA recordings
        # Times are in units of seconds

        arrays_to_use = context_arrays
        assert len(spike_times) == 192, "Expected 192 units"
        for t in range(len(move_begins)):
            # if not is_valid[t]:
            #     continue # we subset now
            if t > 0 and spike_intervals[t][0] < spike_intervals[t-1][1]:
                logger.warning(f"Observation interval for trial {t} overlaps with previous trial, skipping...")
                continue

            start, end = move_begins[t] - cfg.churchland_maze.pretrial_time_s, move_ends[t] + cfg.churchland_maze.posttrial_time_s
            if start <= spike_intervals[t][0]:
                logger.warning("Movement begins before observation interval, cropping...")
                start = spike_intervals[t][0]
            if end > spike_intervals[t][1]:
                if not end_time_mapped[t]: # will definitely be true if end time mapped
                    logger.warning(f"Movement ends after observation interval, cropping... (diff = {(end - spike_intervals[t][1]):.02f}s)")
                end = spike_intervals[t][1]
            if math.isnan(end - start):
                logger.warning(f"Trial {t} has NaN duration, skipping...") # this occurs irreproducibly...
                continue
            time_span = int((end - start) * sampling_rate) + 1 # +1 for rounding error
            trial_spikes = torch.zeros((time_span, len(spike_times)), dtype=torch.uint8)
            for c in range(len(spike_times)):
                if c in drop_units:
                    continue
                unit_times = spike_times.iloc[c]
                unit_times = unit_times[(unit_times >= start) & (unit_times < end)]
                unit_times = unit_times - start
                ms_spike_times, ms_spike_cnt = np.unique(np.floor(unit_times * sampling_rate), return_counts=True)
                trial_spikes[ms_spike_times, c] = torch.tensor(ms_spike_cnt, dtype=torch.uint8)

            # trim to valid length and then reshape
            if cfg.churchland_maze.chop_size_ms > 0:
                trial_spikes = trial_spikes[:cfg.churchland_maze.chop_size_ms]
            trial_spikes = trial_spikes[:cfg.churchland_maze.max_length_ms]
            single_payload = {
                DataKey.spikes: create_spike_payload(trial_spikes, arrays_to_use, cfg=cfg),
            }
            single_path = cache_root / f'{t}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)
