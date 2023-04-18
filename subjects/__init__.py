# from enum import Enum
from ordered_enum import OrderedEnum

r"""
    Dependency notes:
    - We define the enum so there's typing available outside of the tasks module
    - The individual loaders must depend on registry so `register` works
    - The loader registry must depend on the enum so it can be queried
    - To avoid cyclical dependency we must make enum declared before individual loaders
        -  i.e. loader names must be defined in enum rather than enum pulling from loader
"""

class SubjectName(OrderedEnum):
    # We refer to names instead of classes to make converting to singleton pattern easier
    CRS02b = "CRS02b"
    CRS07 = "CRS07"
    BCI02 = "BCI02"
    jenkins = "Jenkins"
    indy = "Indy"
    loco = "Loco"
    nitschke = "Nitschke"
    mihi = "Mihi"
    chewie = "Chewie"
    han = "Han"
    lando = "Lando"
    reggie = "Reggie"
    earl = "Earl"
    nigel = "Nigel"
    rocky = "Rocky"

from .array_info import SubjectInfo, ArrayInfo, ArrayID, GeometricArrayInfo, AliasArrayInfo, SortedArrayInfo
from .array_registry import SubjectArrayRegistry, create_spike_payload
# ? Should we be referencing this instance or the class in calls? IDK
subject_array_registry = SubjectArrayRegistry()

# These import lines ensure registration
from . import pitt_chicago
from . import nlb_monkeys
