from typing import List
from dataclasses import dataclass
import numpy as np

from ..subjects import SubjectName, ArrayInfo, SubjectInfo, GeometricArrayInfo, SubjectArrayRegistry

r"""
    For human BCI experiments at Pitt/Chicago sites.

"""

@dataclass
class PittChicagoArrayInfo(GeometricArrayInfo):
    one_indexed: bool = True
    pedestal_index: int = 0

    def as_indices(self):
        return super().as_indices() + self.pedestal_index * SubjectInfoPittChicago.channels_per_pedestal

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
    @classmethod
    @property
    def arrays(cls):
        return {
            'lateral_s1': cls.sensory_arrays[0],
            'medial_s1': cls.sensory_arrays[1],
            'lateral_m1': cls.motor_arrays[0],
            'medial_m1': cls.motor_arrays[1],
            # We use a simple aliasing system right now, but this abstraction hides the component arrays
            # Which means the data must be stored in groups, since we cannot reconstruct the aliased.
            # To do this correctly, we need to
        }

    @classmethod
    @property
    def aliases(cls):
        return {
            'sensory': ['lateral_s1', 'medial_s1'],
            'motor': ['lateral_m1', 'medial_m1'],
            'all': ['lateral_s1', 'medial_s1', 'lateral_m1', 'medial_m1'],
        }

    @classmethod
    @property
    def sensory_arrays_as_stim_channels(self):
        # -64 because sensory arrays typically start in bank C (channel 65)
        return [arr - 64 for arr in self.sensory_arrays]
    # def get_expected_channel_count(self):
        # return len(self.motor_arrays) * self.channels_per_pedestal

    @classmethod
    def get_subset_channels(cls, pedestals=[], use_motor=True, use_sensory=True) -> np.ndarray:
        # ! Override if needed. Pedestal logic applies for Pitt participants.
        if not pedestals:
            pedestals = range(len(cls.sensory_arrays))
        channels = []
        for p in pedestals:
            if use_motor:
                channels.append(cls.motor_arrays[p].array.flatten() + p * cls.channels_per_pedestal)
            if use_sensory:
                channels.append(cls.sensory_arrays[p].array.flatten() + p * cls.channels_per_pedestal)
        channels = np.concatenate(channels)
        channels = channels[~np.isnan(channels)]
        return np.sort(channels)

    @classmethod
    def get_sensory_record(cls, pedestals=[]):
        if not pedestals:
            pedestals = range(len(cls.sensory_arrays))
        sensory_indices = np.concatenate([
            cls.sensory_arrays[i].flatten() + i * cls.channels_per_pedestal \
                for i in pedestals
        ])
        sensory_indices = sensory_indices[~np.isnan(sensory_indices)]
        return np.sort(sensory_indices)

    @classmethod
    def get_blacklist_channels(cls, flatten=True):
        # ! Override if needed. Pedestal logic only applies for Pitt participants.
        if flatten:
            return cls.blacklist_channels + cls.blacklist_pedestals * cls.channels_per_pedestal
        else:
            return cls.blacklist_channels, cls.blacklist_pedestals

@SubjectArrayRegistry.register
class CRS02b(SubjectInfoPittChicago):
    # Layout shared across motor channels
    name = SubjectName.CRS02b
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
        PittChicagoArrayInfo(array=_motor_layout),
        PittChicagoArrayInfo(array=_motor_layout, pedestal_index=1)
    ]

    # NB: The last 8 even numbered channels are not wired (but typically recorded with the others to form a full 128 block)
    # Don't actually need blacklisting mechanisms since only indexed channels are extracted
    # blacklist_channels = np.array([113, 115, 117, 119, 121, 123, 125, 127]) + 1
    # blacklist_pedestals = np.zeros(8, dtype=int)

    sensory_arrays = [
        PittChicagoArrayInfo(array=np.array([ # Lateral (Anterior), wire bundle to right, viewing from pad side (electrodes down).
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

        PittChicagoArrayInfo(array=np.array([ # Medial (Posterior) wire bundle to right, viewing from pad side (electrodes down)
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
    name = SubjectName.CRS07
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
        PittChicagoArrayInfo(array=_motor_layout),
        PittChicagoArrayInfo(array=_motor_layout, pedestal_index=1)
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
        PittChicagoArrayInfo(array=_sensory_layout),
        PittChicagoArrayInfo(array=_sensory_layout, pedestal_index=1)
    ]

@SubjectArrayRegistry.register
class BCI02(SubjectInfoPittChicago):
    name = SubjectName.BCI02
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
        PittChicagoArrayInfo(array=_motor_layout), # * BCI02's entire medial array is disabled
        PittChicagoArrayInfo(array=_motor_layout, pedestal_index=1)
    ]

    sensory_arrays = [
        PittChicagoArrayInfo(array=np.array([ # Medial Sensory
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

        PittChicagoArrayInfo(array=np.array([ # LateralSensory
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

    @classmethod
    @property # Order flipped for BCI02
    def arrays(cls):
        return {
            'lateral_s1': cls.sensory_arrays[1],
            'medial_s1': cls.sensory_arrays[0],
            'lateral_m1': cls.motor_arrays[1],
            'medial_m1': cls.motor_arrays[0],
        }


