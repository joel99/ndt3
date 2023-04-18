from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Type, Tuple, Optional, Union
import numpy as np

from ..subjects import SubjectName

ArrayID = str
@dataclass
class ArrayInfo:
    is_exact: bool = False # is_exact becomes synonymous with wheter we can query for indices

    @abstractmethod
    def get_channel_count(self) -> int:
        raise NotImplementedError


@dataclass
class GeometricArrayInfo(ArrayInfo):
    r"""
        Contains some metadata.
        There is no way to blacklist indices on the fly currently, e.g. if there's an issue in a particular dataset.
        array: array of electrode indices, possibly reflects the geometry of the implanted array.
    """

    array: np.ndarray = np.array([])
    is_exact: bool = True
    one_indexed: bool = False

    def as_indices(self):
        r"""
            return indices where this array's data is typically stored in a broader dataset.
        """
        indices = self.array[~np.isnan(self.array)].flatten()
        if self.one_indexed:
            indices = indices - 1
        return indices

    def get_channel_count(self):
        return ~np.isnan(self.array).sum()


@dataclass
class SortedArrayInfo(ArrayInfo):
    r"""
        Sorted arrays unfortunately have no consistent interface from session to session.
        We support a simple interface; really readin logic (in `model.__init__`) needs to be queried from session info if we want per session layer
    """
    is_exact: bool = False # count is not exact
    _max_channels: int = 140
    def get_channel_count(self):
        return self._max_channels

@dataclass
class AliasArrayInfo(ArrayInfo):
    # mock some attrs
    arrays: List[ArrayInfo] = field(default_factory=list)

    def __init__(self, *arrays: List[ArrayInfo]):
        self.arrays = arrays

    def get_channel_count(self):
        return sum([a.get_channel_count() for a in self.arrays])

class SubjectInfo:
    r"""
        Right now, largely a wrapper for potentially capturing multiple arrays.
        TODO Should be refactored into a singleton, but right now cheating our way through it
        using just class methods. This is not good python...

        ArrayInfo in turn exists largely to report array shape.
        Provides info relating channel ID (1-indexed, a pedestal/interface concept) to location _within_ array (a brain tissue concept).
        This doesn't directly provide location _within_ system tensor, which might have multi-arrays without additional logic! (see hardcoded constants in this class)
        Agnostic to bank info.
    """
    name: SubjectName # This is typically redundant with __name__, but JY doesn't know how to pythonically avoid stating this
    # without pulling out a full singleton.

    # 1 indexed channel in pedestal, for each pedestal with recordings.
    # This is mostly consistent across participants, but with much hardcoded logic...
    # By convention, stim channels 1-32 is anterior 65-96, and 33-64 is posterior 65-96.
    # Is spatially arranged to reflect true geometry of array.
    # ! Note - this info becomes desynced once we subset channels. We can probably keep it synced by making this class track state, but that's work for another day...

    # Subjects hold onto arrays and aliases without subject tags (no particular reason)
    # These tags are bound to subject ID via wrap array call
    _arrays: Dict[ArrayID, ArrayInfo] = {} # Registry of array names. ! Should include subject name as well? For easy lookup?
    _aliases: Dict[ArrayID, List[str]] = {} # Refers to other arrays (potentially multiple) used for grouping

    @classmethod
    @property
    def arrays(cls) -> Dict[ArrayID, ArrayInfo]:
        return cls._arrays

    @classmethod
    @property
    def aliases(cls) -> Dict[ArrayID, List[ArrayID]]:
        return cls._aliases

    def get_channel_count(self, arrays: Union[ArrayID, List[ArrayID]] = ""):
        if isinstance(arrays, str) and arrays:
            arrays = [arrays]
        queried = self.arrays.values() if not arrays else [self.arrays[a] for a in arrays]
        return sum([a.get_channel_count() for a in queried])

    @classmethod
    def wrap_array(cls, array_id: ArrayID):
        return f"{cls.name.value}-{array_id}"

    @classmethod
    def unwrap_array(cls, array_id: str):
        return array_id.split('-')[-1]

    @classmethod
    def has_array(cls, array_id: ArrayID, unwrapped=True): # unwrapped
        if unwrapped:
            return array_id in cls.arrays
        return array_id.split('-')[-1] in cls.arrays