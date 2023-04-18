from hydra.core.config_store import ConfigStore
from .config_base import *
from . import presets

cs = ConfigStore.instance()
cs.store(name="config", node=RootConfig)