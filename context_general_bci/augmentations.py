from typing import Dict
import random
import torch
from context_general_bci.config import DataKey


def apply_crop(tensor, start_time, crop_length): # Assumes axis 0
    return tensor[start_time:start_time + crop_length]

def rand_crop_time(raw_payload: Dict[DataKey, torch.Tensor], min_frac=0.4):
    # randomly sample a length >= min_frac * time_length, and then a start time
    aug_payload = {}
    time_length = None
    aug_spike = {}

    for arr in raw_payload[DataKey.spikes]:
        if time_length is None:
            time_length = raw_payload[DataKey.spikes][arr].shape[0]
            crop_length = random.randint(int(min_frac * time_length), time_length)
            start_time = random.randint(0, time_length - crop_length)

        aug_spike[arr] = apply_crop(raw_payload[DataKey.spikes][arr], start_time, crop_length)

    aug_payload[DataKey.spikes] = aug_spike

    for key, val in raw_payload.items():
        if key == DataKey.spikes:
            continue
        if val.shape[0] == time_length:
            aug_payload[key] = apply_crop(val, start_time, crop_length)

    return aug_payload

augmentations = {
    'rand_crop_time': rand_crop_time,
}
