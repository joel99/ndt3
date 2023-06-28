from typing import Dict, List

from enum import Enum
from pathlib import Path
import pandas as pd

from context_general_bci.config import DatasetConfig
from context_general_bci.subjects import SubjectInfo
from context_general_bci.tasks import ExperimentalTask

r"""
    Super light wrapper to define task/loader interface.
"""
class ExperimentalTaskLoader:
    name: ExperimentalTask
    @classmethod
    def load(cls,
        dataset_path: Path,
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        context_arrays: List[str],
        dataset_alias: str,
        task: ExperimentalTask,
    ) -> pd.DataFrame:
        r"""
            Load data from `path` into a dataframe.
            `cfg` contains information about how to load the data.
            (and should contain a sub-config corresponding to any registered task)
            `cache_root` is the root directory for caching single trials.

            Each loader should be responsible for loading/caching all information in paths
        """
        raise NotImplementedError

class ExperimentalTaskRegistry:
    _instance = None
    _loaders: Dict[ExperimentalTask, ExperimentalTaskLoader] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register_manual(cls, name: ExperimentalTask, to_register: ExperimentalTaskLoader):
        cls._loaders[name] = to_register

    @classmethod
    def register(cls, to_register: ExperimentalTaskLoader):
        def wrap(to_register: ExperimentalTaskLoader):
            cls._loaders[to_register.name] = to_register
            return to_register
        return wrap(to_register)

    @classmethod
    def get_loader(cls, name: ExperimentalTask) -> ExperimentalTaskLoader:
        if name not in cls._loaders:
            raise ValueError(f'Loader for {name} not found.')
        return cls._loaders[name]
    # We cannot make the loader query by Enum because the enum is defined by loader attrs (which is a design decision)
