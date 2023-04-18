from typing import List, Dict, Type, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn

from einops import rearrange, reduce
from ..config import DatasetConfig
from ..subjects import SubjectName, SubjectInfo, ArrayID, ArrayInfo, AliasArrayInfo, GeometricArrayInfo

class SubjectArrayRegistry:
    instance = None
    _subject_registry: Dict[SubjectName, SubjectInfo] = {}
    _array_registry: Dict[ArrayID, ArrayInfo] = {}
    _alias_registry: Dict[ArrayID, List[str]] = {}

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    # Pattern taken from habitat.core.registry, but without optional naming
    @classmethod
    def register(cls, to_register: SubjectInfo, assert_type = SubjectInfo):
        def wrap(to_register: SubjectInfo):
            if assert_type is not None:
                assert issubclass(
                    to_register, assert_type
                ), "{} must be a subclass of {}".format(
                    to_register, assert_type
                )
            cls._subject_registry[to_register.name] = to_register # ? Possibly should refer to singleton instance explicitly

            for array in to_register.arrays:
                cls._array_registry[to_register.wrap_array(array)] = to_register.arrays[array]
            for alias in to_register.aliases:
                cls._alias_registry[to_register.wrap_array(alias)] = [to_register.wrap_array(a) for a in to_register.aliases[alias]]
            return to_register
        return wrap(to_register)

    @classmethod
    def resolve_alias(cls, alias: ArrayID) -> List[ArrayID]:
        if alias in cls._array_registry:
            return [alias]
        return cls._alias_registry[alias]

    @classmethod
    def query_by_array(cls, id: ArrayID) -> ArrayInfo:
        if id in cls._array_registry:
            return cls._array_registry[id]
        elif id in cls._alias_registry:
            return AliasArrayInfo(cls._array_registry[a] for a in cls._alias_registry[id])

    @classmethod
    def query_by_array_geometric(cls, id: ArrayID) -> GeometricArrayInfo:
        query = cls.query_by_array(id)
        assert isinstance(query, GeometricArrayInfo), f"{id} is not a geometric array"
        return query

    @classmethod
    def query_by_subject(cls, id: Union[SubjectName, str]) -> SubjectInfo:
        if isinstance(id, str):
            id = getattr(SubjectName, id)
        return cls._subject_registry[id]


def create_spike_payload(spikes: Union[torch.Tensor, np.ndarray], arrays_to_use: List[str], cfg: Optional[DatasetConfig] = None, spike_bin_size_ms=1) -> Dict[str, torch.Tensor]:
    r"""
        spikes: full (dense) array from which to extract recording array structure; Time x Channels (x 1/features)
    """
    spikes = torch.as_tensor(spikes, dtype=torch.uint8)
    if cfg:
        assert cfg.bin_size_ms % spike_bin_size_ms == 0
        bin_factor = cfg.bin_size_ms // spike_bin_size_ms
        # crop first bit of trial to round off
        spikes = spikes[len(spikes) % bin_factor:]
        spikes = reduce(
            spikes, '(t bin) c -> t c 1', bin=bin_factor, reduction='sum'
        )
    elif spikes.ndim == 2:
        spikes = rearrange(spikes, 't c -> t c ()')
    spike_payload = {}

    for a in arrays_to_use:
        array = SubjectArrayRegistry.query_by_array(a)
        if array.is_exact:
            array = SubjectArrayRegistry.query_by_array_geometric(a)
            spike_payload[a] = spikes[:, array.as_indices()].clone()
        else:
            assert len(arrays_to_use) == 1, "Can't use multiple arrays with non-exact arrays"
            spike_payload[a] = spikes.clone()
    return spike_payload

# ==== Defunct


def get_channel_pedestal_and_location(
    channel_ids: np.ndarray, # either C or C x 2 (already has pedestal info)
    subject,
    normalize_space=True,
    mode="record",
    array_label=None, # ! Config should ideally specify what array is used for stim and record, but this API is overkill for now
):
    assert mode in ["record", "stim"], f"{mode} location extraction not known"
    array_cls = SubjectArrayRegistry.query_by_subject(subject)

    print(f"Info: Extracting {mode} array locations")
    def extract_pedestal_and_loc_within_array(one_indexed_channel_ids: np.ndarray, group_size: int):
        if one_indexed_channel_ids.ndim == 2:
            return one_indexed_channel_ids[:, 0], one_indexed_channel_ids[:, 1]
        return (
            ((one_indexed_channel_ids - 1) % group_size) + 1, # channels should be one-indexed
            ((one_indexed_channel_ids - 1) // group_size).astype(int), # pedestals needn't/shouldn't be
        )
    if mode == "stim":
        channels, pedestals = extract_pedestal_and_loc_within_array(channel_ids, array_cls.channels_per_stim_bank)
    else:
        channels, pedestals = extract_pedestal_and_loc_within_array(channel_ids, array_cls.channels_per_pedestal)

    # TODO - get the right array
    if mode == "record":
        arrays = [*zip(array_cls.motor_arrays, array_cls.sensory_arrays)]
    elif mode == "stim":
        arrays = [*zip(array_cls.sensory_arrays_as_stim_channels)]

    spatial_locs = []
    # Works fine for stim
    # For record - if channel is in stim array, then check sensory array. Else
    def get_coordinates_within_any_array(id, arrays: Tuple[np.ndarray]):
        # arrays are assumed to have mutually exclusive entries
        for arr in arrays:
            matches = torch.tensor(arr == id).nonzero()
            if len(matches):
                return matches[0]
        raise Exception(f"Channel ID {id} not found in any candidate array.")
    for channel, pedestal in zip(channels, pedestals):
        spatial_locs.append(get_coordinates_within_any_array(channel, arrays[pedestal]))
    spatial_locs = torch.stack(spatial_locs, 0)

    if normalize_space:
        _long_side = max([max(*array.shape) for array in array_cls.all_arrays])
        spatial_locs = (spatial_locs.float() - _long_side / 2.) / _long_side

    return torch.tensor(pedestals, dtype=torch.int), spatial_locs

class DummyEmbedding(nn.Module):
    def __init__(self, channel_ids: torch.Tensor, *args, **kwargs):
        super().__init__()
        self.register_buffer('embed', torch.zeros((len(channel_ids), 0), dtype=torch.float), 0)
        self.n_out = 0

    def forward(self):
        return self.embed

class ChannelEmbedding(nn.Module):
    def __init__(
        self,
        channel_ids: torch.Tensor, # ints
        subject: str,
        mode="record", # TODO what does this do?
        array_label=None # TODO what does this do?. Also, what is this?
    ):
        super().__init__()

        # embeds channels by ID (fxn shouldn't discriminate b/n stim/recording, but do use separate weights for ID-ing respective fxn)
        # * Note, this should be embedded st we can generalize to unseen channels
        # e.g. we shouldn't just throw in the ID
        PEDESTAL_FEATS = 4
        SPATIAL_FEATS = 2 # 2D Location

        pedestals, self.spatial_embeddings = get_channel_pedestal_and_location(
            channel_ids,
            subject,
            normalize_space=True,
            mode=mode,
            array_label=array_label
        )

        # Note, locations within a pedestal are embedding uniformly; network can transform it later if it needs to
        num_pedestals = len(np.unique(pedestals))

        # Option 1 - something about gradients not getting cleared here...
        # pedestal_embedder = nn.Embedding(num_pedestals, PEDESTAL_FEATS)
        # self.pedestal_embeddings = pedestal_embedder(pedestals) # N -> N x PEDESTAL_FEATS
        # Option 1.5 - same issue
        # pedestal_embedder = nn.Parameter(torch.zeros(num_pedestals, PEDESTAL_FEATS))
        # self.pedestal_embeddings = pedestal_embedder[pedestals.long()] # N -> N x PEDESTAL_FEATS

        # Option 2
        self.register_buffer("pedestals", pedestals)
        self.pedestal_embedder = nn.Embedding(num_pedestals, PEDESTAL_FEATS)

        self.n_out = PEDESTAL_FEATS + SPATIAL_FEATS

    # TODO optimize so I don't forward call by manually initing a bunch of pedestal embeddings
    # If nn.embedding is the problem
    def forward(self):
        pedestal_embeddings = self.pedestal_embedder(self.pedestals) # C -> C x PEDESTAL_FEATS
        self.spatial_embeddings = self.spatial_embeddings.to(pedestal_embeddings.device)
        # self.spatial_embeddings = self.spatial_embeddings.to(self.pedestal_embeddings.device)

        return torch.cat([
            # self.pedestal_embeddings, self.spatial_embeddings # C x N
            pedestal_embeddings, self.spatial_embeddings # C x N
        ], -1)


