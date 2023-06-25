r"""
Wandb helpers - interaction of wandb API and local files
"""
from typing import NamedTuple, Union, Dict, List, Tuple, Any, Optional
from typing import get_type_hints, get_args
from pathlib import Path
import os.path as osp
import numpy as np

import wandb

from context_general_bci.config import RootConfig

def wandb_query_experiment(
    experiment: Union[str, List[str]],
    wandb_user="joelye9",
    wandb_project="context_general_bci",
    order='created_at',
    **kwargs,
):
    if not isinstance(experiment, list):
        experiment = [experiment]
    api = wandb.Api()
    filters = {
        'config.experiment_set': {"$in": experiment},
        **kwargs
    }
    runs = api.runs(f"{wandb_user}/{wandb_project}", filters=filters, order=order)
    return runs

def get_best_ckpt_in_dir(ckpt_dir: Path, tag="val_loss", higher_is_better=False):
    if 'bps' in tag:
        higher_is_better = True
    # Newest is best since we have early stopping callback, and modelcheckpoint only saves early stopped checkpoints (not e.g. latest)
    res = sorted(ckpt_dir.glob("*.ckpt"), key=osp.getmtime)
    res = [r for r in res if tag in r.name]
    if tag:
        # names are of the form {key1}={value1}-{key2}={value2}-...-{keyn}={valuen}.ckpt
        # write regex that parses out the value associated with the tag key
        values = []
        for r in res:
            start = r.stem.find(f'{tag}=')
            end = r.stem.find('-', start+len(tag)+2) # ignore negative
            if end == -1:
                end = len(r.stem)
            values.append(float(r.stem[start+len(tag)+1:end].split('=')[-1]))
        if higher_is_better:
            return res[np.argmax(values)]
        return res[np.argmin(values)]
    return res[-1] # default to newest

# TODO update these
def wandb_query_latest(
    name_kw,
    wandb_user='joelye9',
    wandb_project='context_general_bci',
    exact=False,
    allow_running=False,
    use_display=False, # use exact name
    **filter_kwargs
) -> List[Any]: # returns list of wandb run objects
    # One can imagine moving towards a world where we track experiment names in config and query by experiment instead of individual variants...
    # But that's for the next project...
    # Default sort order is newest to oldest, which is what we want.
    api = wandb.Api()
    target = name_kw if exact else {"$regex": name_kw}
    states = ["finished", "crashed", "failed"]
    if allow_running:
        states.append("running")
    filters = {
        "display_name" if use_display else "config.tag": target,
        "state": {"$in": states}, # crashed == timeout
        **filter_kwargs
    }
    runs = api.runs(
        f"{wandb_user}/{wandb_project}",
        filters=filters
    )
    return runs

def wandb_query_several(
    strings,
    min_time=None,
    latest_for_each_seed=True,
):
    runs = []
    for s in strings:
        runs.extend(wandb_query_latest(
            s, exact=True, latest_for_each_seed=latest_for_each_seed,
            created_at={
                "$gt": min_time if min_time else "2022-01-01"
                }
            ,
            allow_running=True # ! NOTE THIS
        ))
    return runs

def get_best_ckpt_from_wandb_id(
        wandb_project,
        wandb_id,
        tag = "val_loss"
    ):
    wandb_id = wandb_id.split('-')[-1]
    ckpt_dir = Path('./data/runs/') / wandb_project / wandb_id / "checkpoints" # curious, something about checkpoint dumping isn't right
    return get_best_ckpt_in_dir(ckpt_dir, tag=tag)

def get_wandb_run(wandb_id, wandb_project='context_general_bci', wandb_user="joelye9"):
    wandb_id = wandb_id.split('-')[-1]
    api = wandb.Api()
    return api.run(f"{wandb_user}/{wandb_project}/{wandb_id}")


r"""
    For experiment auto-inheritance.
    Look in wandb lineage with pointed experiment set for a run sharing the tag. Use that run's checkpoint.
"""
def get_wandb_lineage(cfg: RootConfig):
    assert cfg.inherit_exp, "Must specify experiment set to inherit from"
    api = wandb.Api()
    lineage_query = cfg.tag
    if cfg.inherit_tag:
        # Find the unannotated part of the tag and substitute inheritance
        # (hardcoded)
        lineage_pieces = lineage_query.split('-')
        lineage_query = '-'.join([cfg.inherit_tag] + lineage_pieces[1:])
    if 'sweep' in lineage_query:
        # find sweep and truncate
        lineage_query = lineage_query[:lineage_query.find('sweep')-1] # - m-dash

    # specific patch for any runs that need it...
    additional_filters = {}
    runs = api.runs(
        f"{cfg.wandb_user}/{cfg.wandb_project}",
        filters={
            "config.experiment_set": cfg.inherit_exp,
            "config.tag": lineage_query,
            "state": {"$in": ["finished", "running", "crashed", 'failed']},
            **additional_filters
        }
    )
    if len(runs) == 0:
        raise ValueError(f"No wandb runs found for experiment set {cfg.inherit_exp} and tag {lineage_query}")
    # Basic sanity checks on the loaded checkpoint
    # check runtime
    # Allow crashed, which is slurm timeout
    if runs[0].state != 'crashed' and runs[0].summary.get("_runtime", 0) < 1 * 60: # (seconds)
        raise ValueError(f"InheritError: Run {runs[0].id} abnormal runtime {runs[0].summary.get('_runtime', 0)}")
    if runs[0].state == 'failed':
        print(f"Warning: InheritError: Initializing from failed {runs[0].id}, likely due to run timeout. Indicates possible sub-convergence.")

    return runs[0] # auto-sorts to newest

def wandb_run_exists(cfg: RootConfig, experiment_set: str="", tag: str="", other_overrides: Dict[str, Any] = {}, allowed_states=["finished", "running", "crashed", "failed"]):
    r"""
        Intended to do be used within the scope of an auto-launcher.
        Only as specific as the overrides specify, will be probably too liberal with declaring a run exists if you don't specify enough.
    """
    if not cfg.experiment_set:
        return False
    api = wandb.Api()
    print(other_overrides)
    if 'init_from_id' in other_overrides:
        del other_overrides['init_from_id'] # oh jeez... we've been rewriting this in run.py and doing redundant runs because we constantly query non-inits
    runs = api.runs(
        f"{cfg.wandb_user}/{cfg.wandb_project}",
        filters={
            "config.experiment_set": experiment_set if experiment_set else cfg.experiment_set,
            "config.tag": tag if tag else cfg.tag,
            "state": {"$in": allowed_states},
            **other_overrides,
        }
    )
    return len(runs) > 0