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
class ExperimentalTask(OrderedEnum):
    passive_icms = "passive_icms"
    nlb_maze = "nlb_maze"
    nlb_rtt = "nlb_rtt"
    churchland_maze = "churchland_maze"
    churchland_misc = "churchland_misc"
    odoherty_rtt = "odoherty_rtt"
    dyer_co = "dyer_co"
    gallego_co = "gallego_co"
    pitt_co = "pitt_co"
    observation = "observation"
    ortho = "ortho"
    fbc = "fbc"
    unstructured = "unstructured" # Pitt free play
    delay_reach = "delay_reach"

    marino_batista_mp_bci = "marino_batista_mp_bci"
    marino_batista_mp_reaching = "marino_batista_mp_reaching"
    marino_batista_mp_iso_force = "marino_batista_mp_iso_force"

from .task_registry import ExperimentalTaskRegistry, ExperimentalTaskLoader
# Exports
from .passive_icms import ICMSLoader
from .nlb import MazeLoader, RTTLoader
from .rtt import ODohertyRTTLoader
from .maze import ChurchlandMazeLoader
from .myow_co import DyerCOLoader
from .gallego_co import GallegoCOLoader
from .churchland_misc import ChurchlandMiscLoader
from .pitt_co import PittCOLoader
from .delay_reach import DelayReachLoader
from .marino_batista import MarinoBatistaLoader