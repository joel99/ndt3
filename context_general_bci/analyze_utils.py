# Miscellany
from typing import NamedTuple, Union, Dict, List, Tuple, Any, Optional
from typing import get_type_hints, get_args
from collections import defaultdict
from copy import deepcopy
import re
from enum import Enum
import os.path as osp
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
import wandb
import scipy.signal as signal
import torch
from torch.utils.data import DataLoader
import itertools
from dacite import from_dict

from context_general_bci.utils import get_best_ckpt_from_wandb_id
from context_general_bci.model import BrainBertInterface, load_from_checkpoint
from context_general_bci.dataset import DataAttrs, SpikingDataset
from context_general_bci.config import RootConfig

WandbRun = Any
import seaborn as sns

STYLEGUIDE = {
    "palette": sns.color_palette('colorblind', 5),
    "hue_order": ['single', 'session', 'subject', 'task'],
    "markers": {
        'single': 'o',
        'session': 'D',
        'subject': 's',
        'task': 'X',
    }
}

def cast_paths_and_enums(cfg: Dict, template=RootConfig()):
    # recursively cast any cfg field that is a path in template to a path, since dacite doesn't support our particular case quite well
    # thinking about it more - the weak link is wandb; which casts enums and paths to __str__
    # and now we have to recover from __str__
    def search_enum(str_rep: str, enum: Enum):
        for member in enum:
            if str_rep == str(member):
                return member
        raise ValueError(f"Could not find {str_rep} in {enum}")
    for k, v in get_type_hints(template).items():
        if v == Any: # optional values
            continue
        elif k not in cfg:
            continue # new attr
        elif v == Path:
            cfg[k] = Path(cfg[k])
        elif isinstance(cfg[k], list):
            for i, item in enumerate(cfg[k]):
                generic = get_args(v)[0]
                if issubclass(generic, Enum):
                    cfg[k][i] = search_enum(item, generic)
        elif issubclass(v, Enum):
            cfg[k] = search_enum(cfg[k], v)
        elif isinstance(cfg[k], dict):
            # print(f"recursing with {k}")
            cast_paths_and_enums(cfg[k], template=v)
    return cfg

def create_typed_cfg(cfg: Dict) -> RootConfig:
    cfg = cast_paths_and_enums(cfg)
    return from_dict(data_class=RootConfig, data=cfg)

def load_wandb_run(run: WandbRun, tag="val_loss") -> Tuple[BrainBertInterface, RootConfig, DataAttrs]:
    run_data_attrs = from_dict(data_class=DataAttrs, data=run.config['data_attrs'])
    config_copy = deepcopy(run.config)
    del config_copy['data_attrs']
    cfg: RootConfig = OmegaConf.create(create_typed_cfg(config_copy)) # Note, unchecked cast, but we often fiddle with irrelevant variables and don't want to get caught up
    ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, run.id, tag=tag)
    print(f"Loading {ckpt}")
    model = load_from_checkpoint(ckpt)
    # model = BrainBertInterface.load_from_checkpoint(ckpt, cfg=cfg)
    return model, cfg, run_data_attrs

WandbRun = Any
def get_run_config(run: WandbRun, tag="val_loss"):
    run_data_attrs = from_dict(data_class=DataAttrs, data=run.config['data_attrs'])
    del run.config['data_attrs']
    cfg: RootConfig = OmegaConf.create(create_typed_cfg(run.config)) # Note, unchecked cast, but we often fiddle with irrelevant variables and don't want to get caught up
    return cfg

def get_dataloader(dataset: SpikingDataset, batch_size=100, num_workers=4, **kwargs) -> DataLoader:
    return DataLoader(dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=dataset.collater_factory()
    )

def stack_batch(batch_out: List[Dict[str, torch.Tensor]]):
    all_lists = defaultdict(list)
    for batch in batch_out:
        for k, v in batch.items():
            if isinstance(v, float) or isinstance(v, int):
                v = [v]
            all_lists[k].extend(v)
    out = {}
    for k, v in all_lists.items():
        if isinstance(v[0], torch.Tensor):
            # try stack
            if all(v2.size() == v[0].size() for v2 in v[1:]):
                out[k] = torch.stack(v)
            else:
                out[k] = v
        else:
            out[k] = torch.tensor(v).mean() # some metric
    return out


# Compute Gauss window and std with respect to bins
def gauss_smooth(spikes, bin_size, kernel_sd=0.05):
    # input b t c
    gauss_bin_std = kernel_sd / bin_size
    # the window extends 3 x std in either direction
    win_len = int(7 * gauss_bin_std) # ! Changed from 6 to 7 so there is a peak
    # Create Gaussian kernel
    window = signal.gaussian(win_len, gauss_bin_std, sym=True)
    window /=  np.sum(window)
    window = torch.tensor(window, dtype=torch.float)
    # Record B T C
    b, t, c = spikes.size()
    spikes = spikes.permute(0, 2, 1).reshape(b*c, 1, t)
    # Convolve window (B 1 T) with record as convolution will sum across channels.
    window = window.unsqueeze(0).unsqueeze(0)
    smooth_spikes = torch.nn.functional.conv1d(spikes, window, padding="same")
    return smooth_spikes.reshape(b, c, t).permute(0, 2, 1)


def prep_plt(ax=None, **kwargs):
    if isinstance(ax, np.ndarray):
        for _ax in ax.ravel():
            _prep_plt(_ax, **kwargs)
    else:
        ax = _prep_plt(ax, **kwargs)
    return ax

def _prep_plt(ax=None, spine_alpha=0.3, size="small", big=False):
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    LARGE_SIZE = 15
    if big:
        size = "large"
    if size == "medium":
        SMALL_SIZE = 12
        MEDIUM_SIZE = 16
        LARGE_SIZE = 20
    if size == "large":
        SMALL_SIZE = 18
        # SMALL_SIZE = 20
        MEDIUM_SIZE = 22
        # MEDIUM_SIZE = 24
        LARGE_SIZE = 26
        # LARGE_SIZE = 28
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.style.use('seaborn-muted')
    if ax is None:
        plt.figure(figsize=(6,4))
        ax = plt.gca()
    ax.spines['bottom'].set_alpha(spine_alpha)
    ax.spines['left'].set_alpha(spine_alpha)
    # ax.spines['top'].set_alpha(spine_alpha)
    ax.spines['top'].set_alpha(0)
    # ax.spines['right'].set_alpha(spine_alpha)
    ax.spines['right'].set_alpha(0)
    ax.grid(alpha=0.25)
    return ax


# Transformer hook
# https://gist.github.com/airalcorn2/50ec06517ce96ecc143503e21fa6cb91
# Patch to get attention weights
def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []

r"""
    Usage:
    save_output = SaveOutput()
    patch_attention(transformer.layers[-1].self_attn)
    hook_handle = transformer.layers[-1].self_attn.register_forward_hook(save_output)
"""


class DataManipulator:
    r"""
        Utility class. Refactored out of `ICMSDataset` to keep that focused to dataloading.
    """

    @staticmethod
    def kernel_smooth(
        spikes: torch.Tensor,
        window
    ) -> torch.Tensor:
        window = torch.tensor(window).float()
        window /=  window.sum()
        # Record B T C
        b, t, c = spikes.size()
        spikes = spikes.permute(0, 2, 1).reshape(b*c, 1, t).float()
        # Convolve window (B 1 T) with record as convolution will sum across channels.
        window = window.unsqueeze(0).unsqueeze(0)
        smooth_spikes = torch.nn.functional.conv1d(spikes, window, padding="same")
        return smooth_spikes.reshape(b, c, t).permute(0, 2, 1)

    @staticmethod
    def gauss_smooth(
        spikes: torch.Tensor,
        bin_size: float,
        kernel_sd=0.05,
        window_deviations=7, # ! Changed bandwidth from 6 to 7 so there is a peak
        past_only=False
    ) -> torch.Tensor:
        r"""
            Compute Gauss window and std with respect to bins

            kernel_sd: in seconds
            bin_size: in seconds
            past_only: Causal smoothing, only wrt past bins - we don't expect smooth firing in stim datasets as responses are highly driven by stim.
        """
        # input b t c
        gauss_bin_std = kernel_sd / bin_size
        # the window extends 3 x std in either direction
        win_len = int(window_deviations * gauss_bin_std)
        # Create Gaussian kernel
        window = signal.gaussian(win_len, gauss_bin_std, sym=True)
        if past_only:
            window[len(window) // 2 + 1:] = 0 # Always include current timestep
            # if len(window) % 2:
            # else:
                # window[len(window) // 2 + 1:] = 0
        return DataManipulator.kernel_smooth(spikes, window)


