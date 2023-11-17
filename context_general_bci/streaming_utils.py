from typing import Dict
import logging
import torch

from context_general_bci.config import DataKey, BatchKey

r"""
    Data utilities for mixing streams, should be model agnostic
"""

def precrop_batch(
    batch: Dict[BatchKey, torch.Tensor], # item also works (no batch dimension), due to broadcasting
    crop_timesteps: int,
):
    sanitize = lambda x: x.name if DataKey.time.name in batch else x # stringify - needed while we have weird dataloader misparity
    spike_time = batch[sanitize(DataKey.time)]
    cov_time = batch[sanitize(DataKey.covariate_time)]
    return_time = batch[sanitize(DataKey.task_return_time)]
    constraint_time = batch[sanitize(DataKey.constraint_time)]

    flatten = spike_time.ndim == 2
    if flatten:
        logging.warning("Assuming consistent time across batch")
        spike_time = spike_time[0]
        cov_time = cov_time[0]
        return_time = return_time[0]
        constraint_time = constraint_time[0]
        out = {
            sanitize(DataKey.spikes): batch[sanitize(DataKey.spikes)][:, spike_time < crop_timesteps],
            sanitize(DataKey.time): batch[sanitize(DataKey.time)][:, spike_time < crop_timesteps],
            sanitize(DataKey.position): batch[sanitize(DataKey.position)][:, spike_time < crop_timesteps],
            sanitize(DataKey.bhvr_vel): batch[sanitize(DataKey.bhvr_vel)][:, cov_time < crop_timesteps],
            sanitize(DataKey.covariate_time): batch[sanitize(DataKey.covariate_time)][:, cov_time < crop_timesteps],
            sanitize(DataKey.covariate_space): batch[sanitize(DataKey.covariate_space)][:, cov_time < crop_timesteps],
            sanitize(DataKey.task_reward): batch[sanitize(DataKey.task_reward)][:, return_time < crop_timesteps],
            sanitize(DataKey.task_return): batch[sanitize(DataKey.task_return)][:, return_time < crop_timesteps],
            sanitize(DataKey.task_return_time): batch[sanitize(DataKey.task_return_time)][:, return_time < crop_timesteps],
            sanitize(DataKey.constraint): batch[sanitize(DataKey.constraint)][:, constraint_time < crop_timesteps],
            sanitize(DataKey.constraint_time): batch[sanitize(DataKey.constraint_time)][:, constraint_time < crop_timesteps],
            sanitize(DataKey.constraint_space): batch[sanitize(DataKey.constraint_space)][:, constraint_time < crop_timesteps],

        }
    else:
        out = {
            sanitize(DataKey.spikes): batch[sanitize(DataKey.spikes)][spike_time < crop_timesteps],
            sanitize(DataKey.time): spike_time[spike_time < crop_timesteps],
            sanitize(DataKey.position): batch[sanitize(DataKey.position)][spike_time < crop_timesteps],
            sanitize(DataKey.bhvr_vel): batch[sanitize(DataKey.bhvr_vel)][cov_time < crop_timesteps],
            sanitize(DataKey.covariate_time): cov_time[cov_time < crop_timesteps],
            sanitize(DataKey.covariate_space): batch[sanitize(DataKey.covariate_space)][cov_time < crop_timesteps],
            sanitize(DataKey.task_reward): batch[sanitize(DataKey.task_reward)][return_time < crop_timesteps],
            sanitize(DataKey.task_return): batch[sanitize(DataKey.task_return)][return_time < crop_timesteps],
            sanitize(DataKey.task_return_time): return_time[return_time < crop_timesteps],
            sanitize(DataKey.constraint): batch[sanitize(DataKey.constraint)][constraint_time < crop_timesteps],
            sanitize(DataKey.constraint_time): constraint_time[constraint_time < crop_timesteps],
            sanitize(DataKey.constraint_space): batch[sanitize(DataKey.constraint_space)][constraint_time < crop_timesteps],
        }
    if sanitize(DataKey.covariate_labels) in batch:
        out[sanitize(DataKey.covariate_labels)] = batch[sanitize(DataKey.covariate_labels)]
    return out

def postcrop_batch(
    batch: Dict[BatchKey, torch.Tensor],
    crop_timesteps: int,
):
    # Hm. This will flatten the batch, since there's no guarantees. OK, we'll just squeeze out the time dimension
    sanitize = lambda x: x.name if x.name in batch else x  # stringify
    spike_time = batch[sanitize(DataKey.time)]
    flatten = spike_time.ndim == 2
    cov_time = batch[sanitize(DataKey.covariate_time)]
    return_time = batch[sanitize(DataKey.task_return_time)]
    constraint_time = batch[sanitize(DataKey.constraint_time)]
    if flatten:
        logging.warning("Assuming consistent time across batch")
        spike_time = spike_time[0]
        cov_time = cov_time[0]
        return_time = return_time[0]
        constraint_time = constraint_time[0]
        return {
            sanitize(DataKey.spikes): batch[sanitize(DataKey.spikes)][:, spike_time >= crop_timesteps],
            sanitize(DataKey.time): batch[sanitize(DataKey.time)][:, spike_time >= crop_timesteps] - crop_timesteps,
            sanitize(DataKey.position): batch[sanitize(DataKey.position)][:, spike_time >= crop_timesteps],
            sanitize(DataKey.bhvr_vel): batch[sanitize(DataKey.bhvr_vel)][:, cov_time >= crop_timesteps],
            sanitize(DataKey.covariate_time): batch[sanitize(DataKey.covariate_time)][:, cov_time >= crop_timesteps]  - crop_timesteps,
            sanitize(DataKey.covariate_space): batch[sanitize(DataKey.covariate_space)][:, cov_time >= crop_timesteps],
            sanitize(DataKey.task_reward): batch[sanitize(DataKey.task_reward)][:, return_time >= crop_timesteps],
            sanitize(DataKey.task_return): batch[sanitize(DataKey.task_return)][:, return_time >= crop_timesteps],
            sanitize(DataKey.task_return_time): batch[sanitize(DataKey.task_return_time)][:, return_time >= crop_timesteps]  - crop_timesteps,
            sanitize(DataKey.constraint): batch[sanitize(DataKey.constraint)][:, constraint_time >= crop_timesteps],
            sanitize(DataKey.constraint_time): batch[sanitize(DataKey.constraint_time)][:, constraint_time >= crop_timesteps]  - crop_timesteps,
            sanitize(DataKey.constraint_space): batch[sanitize(DataKey.constraint_space)][:, constraint_time >= crop_timesteps],
            sanitize(DataKey.covariate_labels): batch[sanitize(DataKey.covariate_labels)],
        }
    return {
        sanitize(DataKey.spikes): batch[sanitize(DataKey.spikes)][spike_time >= crop_timesteps],
        sanitize(DataKey.time): spike_time[spike_time >= crop_timesteps],
        sanitize(DataKey.position): batch[sanitize(DataKey.position)][spike_time >= crop_timesteps],
        sanitize(DataKey.bhvr_vel): batch[sanitize(DataKey.bhvr_vel)][cov_time >= crop_timesteps],
        sanitize(DataKey.covariate_time): cov_time[cov_time >= crop_timesteps],
        sanitize(DataKey.covariate_space): batch[sanitize(DataKey.covariate_space)][cov_time >= crop_timesteps],
        sanitize(DataKey.task_reward): batch[sanitize(DataKey.task_reward)][return_time >= crop_timesteps],
        sanitize(DataKey.task_return): batch[sanitize(DataKey.task_return)][return_time >= crop_timesteps],
        sanitize(DataKey.task_return_time): return_time[return_time >= crop_timesteps],
        sanitize(DataKey.constraint): batch[sanitize(DataKey.constraint)][constraint_time >= crop_timesteps],
        sanitize(DataKey.constraint_time): constraint_time[constraint_time >= crop_timesteps],
        sanitize(DataKey.constraint_space): batch[sanitize(DataKey.constraint_space)][constraint_time >= crop_timesteps],
        sanitize(DataKey.covariate_labels): batch[sanitize(DataKey.covariate_labels)],
    }

def prepend_prompt(
    batch_primary,
    prompt, # Assumes batch dim 1, prepended
): # In-place mods
    out = {}
    def batchify(t: torch.Tensor, ref: torch.Tensor): # B x ...
        out_rep = [ref.size(0)] + [1] * (t.dim())
        return t.unsqueeze(0).repeat(out_rep).to(device=ref.device)
    def bind_ref(t: torch.Tensor, ref: torch.Tensor):
        # breakpoint()
        return torch.cat([batchify(t, ref), ref], dim=1)
    time_offset = prompt[DataKey.time].max() + 1
    for k in prompt: # TODO for cleaner code, make reference use .name to begin with
        # breakpoint()
        # print(k)
        if not isinstance(prompt[k], torch.Tensor):
            out[k.name] = prompt[k]
            continue
        if 'time' in k.name:
            out[k.name] = bind_ref(prompt[k], batch_primary[k.name] + time_offset)
        else:
            out[k.name] = bind_ref(prompt[k], batch_primary[k.name])
        if k in [DataKey.task_return, DataKey.task_reward, DataKey.task_return.name, DataKey.task_reward.name]:
            out[k.name] = out[k.name][..., 0] # no hidden dim, TODO not sure why hidden is showing up
    for k in batch_primary:
        if k not in out:
            out[k] = batch_primary[k]
    return out