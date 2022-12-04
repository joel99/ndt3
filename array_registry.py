from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Type, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

# Another registry of sorts - holding subject + array info instead of experimental info.
ArrayID = str
@dataclass
class ArrayInfo:
    r"""
        Contains some metadata.
        There is no way to blacklist indices on the fly currently, e.g. if there's an issue in a particular dataset.
        array: array of electrode indices, possibly reflects the geometry of the implanted array.
    """
    array: np.ndarray

    # Potentially geometric
    geometric: bool = False
    one_indexed: bool = False

    def as_indices(self):
        r"""
            return indices where this array's data is typically stored in a broader dataset.
        """
        indices = self.array[self.array != np.nan].flatten() if self.geometric else self.array
        if self.one_indexed:
            indices = indices - 1
        return indices


    def get_channel_count(self):
        return (self.array != np.nan).sum()

@dataclass
class PittChicagoArrayInfo(ArrayInfo):
    geometric: bool = True
    one_indexed: bool = True
    pedestal_index: int = 0

    def as_indices(self):
        return super().as_indices() + self.pedestal_index * SubjectInfoPittChicago.channels_per_pedestal

class SubjectInfo:
    r"""
        Right now, largely a wrapper for potentially capturing multiple arrays.
        ArrayInfo in turn exists largely to report array shape.
        Provides info relating channel ID (1-indexed, a pedestal/interface concept) to location _within_ array (a brain tissue concept).
        This doesn't directly provide location _within_ system tensor, which might have multi-arrays without additional logic! (see hardcoded constants in this class)
        Agnostic to bank info.
    """
    # 1 indexed channel in pedestal, for each pedestal with recordings.
    # This is mostly consistent across participants, but with much hardcoded logic...
    # By convention, stim channels 1-32 is anterior 65-96, and 33-64 is posterior 65-96.
    # Is spatially arranged to reflect true geometry of array.
    # ! Note - this info becomes desynced once we subset channels. We can probably keep it synced by making this class track state, but that's work for another day...
    _arrays: Dict[ArrayID, ArrayInfo] # Registry of array names. ! Should include subject name as well? For easy lookup?
    _aliases: Dict[ArrayID, List[str]] # Refers to other arrays (potentially multiple) used for grouping

    @property
    def arrays(self) -> Dict[ArrayID, ArrayInfo]:
        return self._arrays

    @property
    def aliases(self) -> Dict[ArrayID, List[ArrayID]]:
        return self._aliases

    def get_channel_count(self, arrays: ArrayID | List[ArrayID] = ""):
        if isinstance(arrays, str) and arrays:
            arrays = [arrays]
        queried = self.arrays.values() if not arrays else [self.arrays[a] for a in arrays]
        return sum([a.get_channel_count() for a in queried])


class SubjectArrayRegistry:
    instance = None
    _subject_registry: Dict[str, SubjectInfo] = {}
    _array_registry: Dict[ArrayID, ArrayInfo] = {}
    _alias_registry: Dict[ArrayID, List[str]] = {}

    def __new__(cls, init_items: List[SubjectInfo]=[]):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
            cls.instance.register(init_items)
        return cls.instance

    @staticmethod
    def wrap_array(cls, subject_id, array_id):
        return f"{subject_id}-{array_id}"

    # Pattern taken from habitat.core.registry
    @classmethod
    def register(cls, to_register: SubjectInfo, name: Optional[str]=None, assert_type = SubjectInfo):
        def wrap(to_register):
            if assert_type is not None:
                assert issubclass(
                    to_register, assert_type
                ), "{} must be a subclass of {}".format(
                    to_register, assert_type
                )
            register_name = to_register.__name__ if name is None else name

            cls._subject_registry[register_name] = to_register # ? Possibly should refer to singleton instance explicitly
            for array in to_register.arrays:
                cls._array_registry[cls.wrap_array(register_name, array)] = to_register.arrays[array]
            for alias in to_register.aliases:
                cls._array_registry[cls.wrap_array(register_name, alias)] = to_register.arrays[alias]
            return to_register
        return wrap(to_register)

    @classmethod
    def resolve_alias(cls, alias: ArrayID) -> List[ArrayID]:
        return cls._alias_registry[alias]

    @classmethod
    def query_by_array(cls, id: ArrayID) -> ArrayInfo:
        if id in cls._array_registry:
            return cls._array_registry[id]
        elif id in cls._alias_registry:
            return ArrayInfo(np.concatenate(cls._array_registry[a].array.flatten() for a in cls._alias_registry[id]))

    @classmethod
    def query_by_subject(cls, id: str) -> SubjectInfo:
        return cls._subject_registry[id]


# ? Should we be referencing this instance or the class in calls? IDK
# ? Preferring the class for now as it provides typing
subject_array_registry = SubjectArrayRegistry()



class SubjectInfoPittChicago(SubjectInfo):
    r"""
        Human array subclass. These folks all have 4 arrays, wired to two pedestals
        ? Is it ok that I don't define `arrays`
        For these human participants, these channels are 1-indexed
    """
    channels_per_pedestal = 128
    channels_per_stim_bank = 32

    motor_arrays: List[ArrayInfo]
    sensory_arrays: List[ArrayInfo]
    # Implicit correspondence - we assume motor_arrays[i] is in same pedestal as sensory_arrays[i]. I'm not sure if this surfaces in any code logic, though.
    blacklist_channels: np.ndarray = np.array([]) # specifies (within pedestal) 1-indexed position of blacklisted channels
    blacklist_pedestals: np.ndarray = np.array([]) # to be paired with above
    # Note that this is not a cross-product, it's paired.

    # Note in general that blacklisting may eventually be turned into a per-session concept, not just a permanent one. Not in the near future.

    @property
    def arrays(self):
        return {
            'lateral_s1': self.sensory_arrays[0],
            'medial_s1': self.sensory_arrays[1],
            'lateral_m1': self.motor_arrays[0],
            'medial_m1': self.motor_arrays[1],
            # We use a simple aliasing system right now, but this abstraction hides the component arrays
            # Which means the data must be stored in groups, since we cannot reconstruct the aliased.
            # To do this correctly, we need to
        }

    @property
    def aliases(self):
        return {
            'sensory': ['lateral_s1', 'medial_s1'],
            'motor': ['lateral_m1', 'medial_m1'],
            'all': ['lateral_s1', 'medial_s1', 'lateral_m1', 'medial_m1'],
        }

    @property
    def sensory_arrays_as_stim_channels(self):
        # -64 because sensory arrays typically start in bank C (channel 65)
        return [arr - 64 for arr in self.sensory_arrays]
    # def get_expected_channel_count(self):
        # return len(self.motor_arrays) * self.channels_per_pedestal

    def get_subset_channels(self, pedestals=[], use_motor=True, use_sensory=True) -> np.ndarray:
        # ! Override if needed. Pedestal logic applies for Pitt participants.
        if not pedestals:
            pedestals = range(len(self.sensory_arrays))
        channels = []
        for p in pedestals:
            if use_motor:
                channels.append(self.motor_arrays[p].flatten() + p * self.channels_per_pedestal)
            if use_sensory:
                channels.append(self.sensory_arrays[p].flatten() + p * self.channels_per_pedestal)
        channels = np.concatenate(channels)
        channels = channels[~np.isnan(channels)]
        return np.sort(channels)

    def get_sensory_record(self, pedestals=[]):
        if not pedestals:
            pedestals = range(len(self.sensory_arrays))
        sensory_indices = np.concatenate([
            self.sensory_arrays[i].flatten() + i * self.channels_per_pedestal \
                for i in pedestals
        ])
        sensory_indices = sensory_indices[~np.isnan(sensory_indices)]
        return np.sort(sensory_indices)

    def get_blacklist_channels(self, flatten=True):
        # ! Override if needed. Pedestal logic only applies for Pitt participants.
        if flatten:
            return self.blacklist_channels + self.blacklist_pedestals * self.channels_per_pedestal
        else:
            return self.blacklist_channels, self.blacklist_pedestals

@SubjectArrayRegistry.register
class CRS02b(SubjectInfoPittChicago):
    # Layout shared across motor channels

    _motor_layout = np.array([ # wire bundle to right, viewing from pad side (electrodes down)
        [np.nan, np.nan, 42, 58, 3, 13, 27, 97, np.nan, np.nan],
        [np.nan, 34, 44, 57, 4, 19, 29, 98, 107, np.nan],
        [33, 36, 51, 62, 7, 10, 31, 99, 108, 117],
        [35, 38, 53, 60, 5, 12, 18, 100, 109, 119],
        [37, 40, 50, 59, 6, 23, 22, 101, 110, 121],
        [39, 43, 46, 64, 9, 25, 24, 102, 111, 123],
        [41, 47, 56, 61, 17, 21, 26, 103, 113, 125],
        [45, 49, 55, 63, 15, 14, 28, 104, 112, 127],
        [np.nan, 48, 54, 2, 8, 16, 30, 105, 115, np.nan],
        [np.nan, np.nan, 52, 1, 11, 20, 32, 106, np.nan, np.nan, ]
    ])
    motor_arrays = [
        PittChicagoArrayInfo(_motor_layout),
        PittChicagoArrayInfo(_motor_layout, pedestal_index=1)
    ]

    # NB: The last 8 even numbered channels are not wired (but typically recorded with the others to form a full 128 block)
    # blacklist_channels = np.array([113, 115, 117, 119, 121, 123, 125, 127]) + 1
    # blacklist_pedestals = np.zeros(8, dtype=int)



    sensory_arrays = [
        PittChicagoArrayInfo(np.array([ # Lateral (Anterior), wire bundle to right, viewing from pad side (electrodes down).
            [65,    np.nan,     72,     np.nan,     85,     91],
            [np.nan,    77, np.nan,         81, np.nan,     92],
            [67,    np.nan,     74,     np.nan,     87, np.nan],
            [np.nan,    79, np.nan,         82, np.nan,     94],
            [69,    np.nan,     76,     np.nan,     88, np.nan],
            [np.nan,    66, np.nan,         84, np.nan,     93],
            [71,    np.nan,     78,     np.nan,     89, np.nan],
            [np.nan,    68, np.nan,         83, np.nan,     96],
            [73,    np.nan,     80,     np.nan,     90, np.nan],
            [75,        70, np.nan,         86, np.nan,     95],
        ])), # - 65 + 1,

        PittChicagoArrayInfo(np.array([ # Medial (Posterior) wire bundle to right, viewing from pad side (electrodes down)
            [65, np.nan, 72, np.nan, 85, 91],
            [np.nan, 77, np.nan, 81, np.nan, 92],
            [67, np.nan, 74, np.nan, 87, np.nan],
            [np.nan, np.nan, np.nan, 82, np.nan, 94],
            [69, 79, 76, np.nan, 88, np.nan],
            [np.nan, 66, np.nan, 84, np.nan, 93],
            [71, np.nan, 78, np.nan, 89, np.nan],
            [np.nan, 68, np.nan, 83, np.nan, 96],
            [73, np.nan, 80, np.nan, 90, np.nan],
            [75, 70, np.nan, 86, np.nan, 95],
        ]), pedestal_index=1) #  - 65 + 33
    ]
    # NB: We don't clone sensory like motor bc there's a small diff

@SubjectArrayRegistry.register
class CRS07(SubjectInfoPittChicago):
    # Layout shared across motor channels

    _motor_layout = np.array([ # wire bundle to right, viewing from pad side (electrodes down)
        [np.nan, 38, 50, 59,  6, 23,  22, 101, 111, np.nan,],
            [33, 40, 46, 64,  9, 25,  24, 102, 113, 128],
            [35, 43, 56, 61, 17, 21,  26, 103, 112, 114],
            [37, 47, 55, 63, 15, 14,  28, 104, 115, 116],
            [39, 49, 54,  2,  8, 16,  30, 105, 117, 118],
            [41, 48, 52,  1, 11, 20,  32, 106, 119, 120],
            [45, 42, 58,  3, 13, 27,  97, 107, 121, 122],
            [34, 44, 57,  4, 19, 29,  99, 108, 123, 124],
            [36, 51, 62,  7, 10, 31,  98, 109, 125, 126],
        [np.nan, 53, 60,  5, 12, 18, 100, 110, 127, np.nan]
    ])
    motor_arrays = [
        PittChicagoArrayInfo(_motor_layout),
        PittChicagoArrayInfo(_motor_layout, pedestal_index=1)
    ]

    _sensory_layout = np.array([ # wire bundle to right, viewing from pad side (electrodes down).
            [65, np.nan, 72, np.nan, 85, 91],
            [np.nan, 77, np.nan, 81, np.nan, 92],
            [67, np.nan, 74, np.nan, 87, np.nan,],
            [np.nan, 79, np.nan, 82, np.nan, 93],
            [69, np.nan, 76, np.nan, 88, np.nan,],
            [np.nan, 66, np.nan, 84, np.nan, 94],
            [71, np.nan, 78, np.nan, 89, np.nan,],
            [np.nan, 68, np.nan, 83, np.nan, 96],
            [73, np.nan, 80, np.nan, 90, np.nan,],
            [75, 70, np.nan, 86, np.nan, 95],
    ])#  - 65 + 1

    sensory_arrays = [
        PittChicagoArrayInfo(_sensory_layout),
        PittChicagoArrayInfo(_sensory_layout, pedestal_index=1)
    ]

@SubjectArrayRegistry.register
class BCI02(SubjectInfoPittChicago):
    # No, the floating point isn't a concern
    _motor_layout = np.array([ # Lat Motor
        [np.nan, 166., 178., 187., 134., 151., 150., 229., 239., np.nan],
        [161., 168., 174., 192., 137., 153., 152., 230., 241., 256.],
        [163., 171., 184., 189., 145., 149., 154., 231., 240., 242.],
        [165., 175., 183., 191., 143., 142., 156., 232., 243., 244.],
        [167., 177., 182., 130., 136., 144., 158., 233., 245., 246.],
        [169., 176., 180., 129., 139., 148., 160., 234., 247., 248.],
        [173., 170., 186., 131., 141., 155., 225., 235., 249., 250.],
        [162., 172., 185., 132., 147., 157., 227., 236., 251., 252.],
        [164., 179., 190., 135., 138., 159., 226., 237., 253., 254.],
        [np.nan, 181., 188., 133., 140., 146., 228., 238., 255., np.nan]
    ])
    motor_arrays = [
        PittChicagoArrayInfo(_motor_layout), # * BCI02's entire medial array is disabled
        PittChicagoArrayInfo(_motor_layout, pedestal_index=1)
    ]

    sensory_arrays = [
        PittChicagoArrayInfo(np.array([ # Medial Sensory
            [65., np.nan, 72., np.nan, 85., 91.],
            [np.nan, 77., np.nan, 81., np.nan, 92.],
            [67., np.nan, 74., np.nan, 87., np.nan],
            [np.nan, 79., np.nan, 82., np.nan, 93.],
            [69., np.nan, 76., np.nan, 88., np.nan],
            [np.nan, 66., np.nan, 84., np.nan, 94.],
            [71., np.nan, 78., np.nan, 89., np.nan],
            [np.nan, 68., np.nan, 83., np.nan, 96.],
            [73., np.nan, 80., np.nan, 90., np.nan],
            [75., 70., np.nan, 86., np.nan, 95.]
        ])), #  - 65 + 1, # Stim channels 1-32

        PittChicagoArrayInfo(np.array([ # LateralSensory
            [193.,  np.nan, 200.,  np.nan, 213., 219.],
            [ np.nan, 205.,  np.nan, 209.,  np.nan, 220.],
            [195.,  np.nan, 202.,  np.nan, 215.,  np.nan],
            [ np.nan, 207.,  np.nan, 210.,  np.nan, 221.],
            [197.,  np.nan, 204.,  np.nan, 216.,  np.nan],
            [ np.nan, 194.,  np.nan, 212.,  np.nan, 222.],
            [199.,  np.nan, 206.,  np.nan, 217.,  np.nan],
            [ np.nan, 196.,  np.nan, 211.,  np.nan, 224.],
            [201.,  np.nan, 208.,  np.nan, 218.,  np.nan],
            [203., 198.,  np.nan, 214.,  np.nan, 223.]
        ]) - 128, pedestal_index=1)#  - 193 + 33 # Stim channels 33-64
    ]

    blacklist_channels = np.concatenate([
        np.arange(1, 129), # 1st motor array
        np.arange(193, 225) # stim channels...
    ])

    blacklist_pedestals = np.zeros(128 + 32, dtype=int)

    @property # Order flipped for BCI02
    def arrays(self):
        return {
            'lateral_s1': self.sensory_arrays[1],
            'medial_s1': self.sensory_arrays[0],
            'lateral_m1': self.motor_arrays[1],
            'medial_m1': self.motor_arrays[0],
        }



# ==== Defunct


def get_channel_pedestal_and_location(
    channel_ids: np.ndarray, # either C or C x 2 (already has pedestal info)
    subject,
    normalize_space=True,
    mode="record",
    array_label=None, # ! Config should ideally specify what array is used for stim and record, but this API is overkill for now
):
    assert mode in ["record", "stim"], f"{mode} location extraction not known"
    array_cls = subject_array_registry.query_by_subject(subject)

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


