import numpy as np
from subjects import SubjectName, SortedArrayInfo, SubjectInfo, SubjectArrayRegistry, GeometricArrayInfo

@SubjectArrayRegistry.register
class Jenkins(SubjectInfo):
    # Churchland maze (NLB)
    name = SubjectName.jenkins
    _arrays = {
        # this is actually just for held-in neurons. `main` is an array used for NLB data
        # technically the data is available to decompose into PMd and M1, but NLB data is sorted
        # so there's no simple map from main to PMd/M1. Hence we make it another pseudo array and hope for the best.
        'main': SortedArrayInfo(_max_channels=137),
        'PMd': GeometricArrayInfo(array=np.arange(96)), # these are _on_ Utah arrays, I just don't have the geometry
        'M1': GeometricArrayInfo(array=np.arange(96) + 96),
    }

@SubjectArrayRegistry.register
class Nitschke(SubjectInfo):
    # Churchland maze + misc, both unsorted two array datasets
    name = SubjectName.nitschke
    _arrays = {
        'PMd': GeometricArrayInfo(array=np.arange(96)),
        'M1': GeometricArrayInfo(array=np.arange(96) + 96),
    }


@SubjectArrayRegistry.register
class Indy(SubjectInfo):
    name = SubjectName.indy
    _arrays = {
        'main': SortedArrayInfo(_max_channels=98), # For NLB
        'M1': GeometricArrayInfo(array=np.arange(96)),
        'S1': GeometricArrayInfo(array=np.arange(96) + 96),
    }

@SubjectArrayRegistry.register
class Loco(SubjectInfo): # RTT https://zenodo.org/record/3854034
    name = SubjectName.loco
    _arrays = {
        'M1': GeometricArrayInfo(array=np.arange(96)),
        'S1': GeometricArrayInfo(array=np.arange(96) + 96),
    }

@SubjectArrayRegistry.register
class Mihi(SubjectInfo):
    name = SubjectName.mihi
    _arrays = {
        'main': SortedArrayInfo(_max_channels=187), # single-session (Dyer)
        'M1': SortedArrayInfo(_max_channels=52), # dual
        'PMd': SortedArrayInfo(_max_channels=121), # dual
    }

@SubjectArrayRegistry.register
class Chewie(SubjectInfo):
    name = SubjectName.chewie
    _arrays = {
        'main': SortedArrayInfo(_max_channels=174), # single-session (Dyer)
        'M1': SortedArrayInfo(_max_channels=88), # left hemisphere
        'PMd': SortedArrayInfo(_max_channels=211), # left hemisphere
    }

@SubjectArrayRegistry.register
class Han(SubjectInfo):
    name = SubjectName.han
    _arrays = {
        'LeftS1Area2': SortedArrayInfo(_max_channels=96), # saw highest 83
    }

@SubjectArrayRegistry.register
class Lando(SubjectInfo):
    name = SubjectName.lando
    _arrays = {
        'LeftS1Area2': SortedArrayInfo(_max_channels=64), # saw highest 46
    }

@SubjectArrayRegistry.register
class Reggie(SubjectInfo):
    name = SubjectName.reggie
    _arrays = {
        'main': GeometricArrayInfo(array=np.arange(94)), # 94 determined by observation
    }