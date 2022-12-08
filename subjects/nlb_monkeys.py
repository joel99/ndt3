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
        'PMd': GeometricArrayInfo(np.arange(96)), # these are _on_ Utah arrays, I just don't have the geometry
        'M1': GeometricArrayInfo(np.arange(96) + 96),
    }

@SubjectArrayRegistry.register
class Nitschke(SubjectInfo):
    # Churchland maze
    name = SubjectName.nitschke
    _arrays = {
        'PMd': GeometricArrayInfo(np.arange(96)),
        'M1': GeometricArrayInfo(np.arange(96) + 96),
    }


@SubjectArrayRegistry.register
class Indy(SubjectInfo):
    name = SubjectName.indy
    _arrays = {
        'main': SortedArrayInfo(_max_channels=137) # TODO
    }