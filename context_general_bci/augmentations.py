from typing import Dict
import torch
from context_general_bci.config import DataKey


def rand_crop_time(data: Dict[DataKey, torch.Tensor]):
    return None

augmentations = {
    'rand_crop_time': rand_crop_time,
}
