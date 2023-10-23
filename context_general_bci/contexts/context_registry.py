# Information about the real world.
# Includes experimental notes, in lieu of readme
# Ideally, this class can be used outside of this specific codebase.

import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
import functools

from .context_info import (
    ContextInfo,
    ReachingContextInfo,
    RTTContextInfo,
    DyerCOContextInfo,
    GallegoCOContextInfo,
    GDrivePathContextInfo,
    BCIContextInfo,
    BatistaContextInfo,
    MillerContextInfo,
    RouseContextInfo
)

from context_general_bci.tasks import ExperimentalTask

CLOSED_LOOP_DIR = 'closed_loop_tests'
r"""
    ContextInfo class is an interface for storing meta-information needed by several consumers, mainly the model, that may not be logged in data from various sources.
    ContextRegistry allows consumers to query for this information from various identifying sources.
    Note - external registry calls should use the instance, not the class.
    This appears to be necessary for typing to work more reliably (unclear).

    Binds subject, task, and other metadata like datapath together.
    JY's current view of  ideal dependency tree is
    Consumer (e.g. model training loop) -- depends -- > ContextRegistry --> Task -> Subject Registry
    - (But currently consumer will dive into some task-specific details, loader could be refactored..)

    Querying examples:
    # context = context_registry.query(alias='odoherty_rtt-Loco')[0] (see formatting of alias in `context_info` defns)
    # datapath = './data/odoherty_rtt/indy_20160407_02.mat'
    # context = context_registry.query_by_datapath(datapath)
"""

r"""
    To support a new task
    - Add a new enum value to ExperimentalTask
    - Add experimental config to DatasetConfig
    - Subclass ContextInfo and implement the abstract methods
"""

class ContextRegistry:
    instance = None
    _registry: Dict[str, ContextInfo] = {}
    search_index = None  # allow multikey querying

    def __new__(cls, init_items: List[ContextInfo]=[]):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
            cls.search_index = pd.DataFrame()
            cls.instance.register(init_items)
        return cls.instance

    def build_search_index(self, items: List[ContextInfo]):
        index = [{
            'id': item.id,
            'task': item.task,
            'datapath': item.datapath.resolve(),
            **item.get_search_index()
        } for item in items]
        return pd.DataFrame(index)

    # ! Note, current pattern is to put all experiments in a big list below; not use this register handle.
    def register(self, context_info: List[ContextInfo | None]):
        context_info = [item for item in context_info if item is not None]
        self.search_index = pd.concat([self.search_index, self.build_search_index(context_info)])
        for item in context_info:
            self._registry[item.id] = item

    def query(self, **search) -> Union[ContextInfo, List[ContextInfo], None]:
        def search_query(df):
            non_str_search = [k for k in search if k != 'alias']
            if non_str_search:
                result = functools.reduce(lambda a, b: a & b, [df[k] == search[k] for k in non_str_search])
            else:
                result = pd.Series(True, index=df.index)
            if 'alias' not in search:
                return result
            if 'alias' not in df:
                print("no alias, nothing registered?")
                return None # test time
            return result & df['alias'].str.contains(search['alias'])

        queried = self.search_index.loc[search_query]
        if len(queried) == 0:
            return None
        elif len(queried) > 1:
            out = [self._registry[id] for id in queried['id']]
            return sorted(out)
        return self._registry[queried['id'].values[0]]

    def query_by_datapath(self, datapath: Union[Path, str]) -> ContextInfo:
        if not isinstance(datapath, Path):
            datapath = Path(datapath)
        found = self.search_index[self.search_index.datapath == datapath.resolve()]
        assert len(found) == 1, f"Found {len(found)} matches for {datapath}"
        return self._registry[found.iloc[0]['id']]

    def query_by_id(self, id: str) -> ContextInfo:
        return self._registry[id]

context_registry = ContextRegistry()

if not os.getenv('NDT_SUPPRESS_DEFAULT_REGISTRY', False):
    context_registry.register([
        *RTTContextInfo.build_several('./data/odoherty_rtt/', alias_prefix='odoherty_rtt'),

        # DyerCOContextInfo.build(('mihi', 1), ExperimentalTask.dyer_co, alias='dyer_co_mihi_1'),
        # DyerCOContextInfo.build(('mihi', 2), ExperimentalTask.dyer_co, alias='dyer_co_mihi_2'),
        DyerCOContextInfo.build(('chewie', 1), ExperimentalTask.dyer_co, alias='dyer_co_chewie_1'),
        DyerCOContextInfo.build(('chewie', 2), ExperimentalTask.dyer_co, alias='dyer_co_chewie_2'),

        *GallegoCOContextInfo.build_from_dir('./data/gallego_co', task=ExperimentalTask.gallego_co),
        # *GallegoCOContextInfo.build_from_dir('./data/gallego_co/dedup', task=ExperimentalTask.gallego_co),

        ReachingContextInfo.build('./data/nlb/000128/sub-Jenkins', ExperimentalTask.nlb_maze, alias='mc_maze'),
        ReachingContextInfo.build('./data/nlb/000138/sub-Jenkins', ExperimentalTask.nlb_maze, alias='mc_maze_large'),
        ReachingContextInfo.build('./data/nlb/000139/sub-Jenkins', ExperimentalTask.nlb_maze, alias='mc_maze_medium'),
        ReachingContextInfo.build('./data/nlb/000140/sub-Jenkins', ExperimentalTask.nlb_maze, alias='mc_maze_small'),
        ReachingContextInfo.build('./data/nlb/000129/sub-Indy', ExperimentalTask.nlb_rtt, alias='mc_rtt'),

        *ReachingContextInfo.build_several('./data/churchland_reaching/000070/sub-Jenkins', ExperimentalTask.churchland_maze, alias_prefix='churchland_maze_jenkins'),
        *ReachingContextInfo.build_several('./data/churchland_reaching/000070/sub-Nitschke', ExperimentalTask.churchland_maze, alias_prefix='churchland_maze_nitschke'),

        # *ReachingContextInfo.build_several('./data/even_chen_delay/000121/sub-JenkinsC', ExperimentalTask.delay_reach, alias_prefix='even_chen_delay_jenkins'),
        *ReachingContextInfo.build_several('./data/delay_reach/000121/sub-Reggie', ExperimentalTask.delay_reach, alias_prefix='even_chen_delay_reggie'),

        *GDrivePathContextInfo.build_from_dir('./data/churchland_misc', blacklist=[]),
        *GDrivePathContextInfo.build_from_dir('./data/churchland_misc', blacklist=['reggie']),

        # *BCIContextInfo.build_from_dir('./data/pitt_co', task_map={
        #     'obs': ExperimentalTask.observation,
        #     'ortho': ExperimentalTask.ortho,
        #     'ortho/fbc': ExperimentalTask.fbc, # when both types are used, opt for more expressive
        #     'fbc': ExperimentalTask.fbc,
        #     'fbc-stitch': ExperimentalTask.fbc,
        #     'unstructured': ExperimentalTask.unstructured,
        #     'free_play': ExperimentalTask.unstructured,
        #     'default': ExperimentalTask.pitt_co,
        #     'unk': ExperimentalTask.pitt_co,
        # }),
        # *BCIContextInfo.build_from_dir(f'./data/{CLOSED_LOOP_DIR}', task_map={
        #     'obs': ExperimentalTask.observation,
        #     'ortho': ExperimentalTask.ortho,
        #     'fbc': ExperimentalTask.fbc,
        # }, alias_prefix='closed_loop_'),
        # *BCIContextInfo.build_from_dir(f'./data/pitt_full', task_map={
        #     'obs': ExperimentalTask.observation,
        #     'ortho': ExperimentalTask.ortho,
        #     'fbc': ExperimentalTask.fbc,
        # }, alias_prefix='pitt_full_'),
        *BCIContextInfo.build_from_dir(f'./data/pitt_broad', task_map={}, alias_prefix='pitt_broad_'),
        # The following are identical datasets, just replicated so I could run different preproc in parallel
        # *BCIContextInfo.build_from_dir(f'./data/pitt_test', task_map={}, alias_prefix='pitt_test_'),
        # *BCIContextInfo.build_from_dir(f'./data/pitt_return', task_map={}, alias_prefix='pitt_return_'),

        # *BCIContextInfo.build_from_dir(f'./data/pitt_misc/mat', task_map={
        #     'obs': ExperimentalTask.observation,
        #     'ortho': ExperimentalTask.ortho,
        #     'fbc': ExperimentalTask.fbc,
        #     'unstructured': ExperimentalTask.unstructured,
        #     'free_play': ExperimentalTask.unstructured,
        # }, alias_prefix='pitt_misc_'), # Don't use in pretraining
        # *BCIContextInfo.build_from_dir_varied('./data/pitt_varied', task_map={
        #     'unstructured': ExperimentalTask.unstructured,
        #     'free_play': ExperimentalTask.unstructured,
        #     'default': ExperimentalTask.pitt_co,
        # }),
        # *MillerContextInfo.build_from_dir('./data/miller/adversarial', task=ExperimentalTask.miller),

        *MillerContextInfo.build_from_dir('./data/miller/Jango_ISO_2015', task=ExperimentalTask.miller),
        *MillerContextInfo.build_from_dir('./data/miller/Spike_ISO_2012', task=ExperimentalTask.miller),
        *MillerContextInfo.build_from_dir('./data/miller/Chewie_CO_2016', task=ExperimentalTask.miller),
        *MillerContextInfo.build_from_dir('./data/miller/Mihili_CO_2014', task=ExperimentalTask.miller),
        *MillerContextInfo.build_from_dir('./data/miller/Mihili_RT_2013_2014', task=ExperimentalTask.miller),
        *MillerContextInfo.build_from_dir('./data/miller/Greyson_Key_2019', task=ExperimentalTask.miller),

        # *MillerContextInfo.build_from_dir('./data/miller/Pop_PG_2021', task=ExperimentalTask.miller), # TODO fixup - it's at 50ms, maybe not worth in near future.

        *RouseContextInfo.build_from_dir('./data/rouse_precision/monk_p/COT_SpikesCombined', task=ExperimentalTask.rouse),
        *RouseContextInfo.build_from_dir('./data/rouse_precision/monk_q/COT_SpikesCombined', task=ExperimentalTask.rouse),

        # *BatistaContextInfo.build_from_dir('./data/marino_batista/earl_multi_posture_isometric_force', task=ExperimentalTask.marino_batista_mp_iso_force),
        # *BatistaContextInfo.build_from_dir('./data/marino_batista/earl_multi_posture_bci', task=ExperimentalTask.marino_batista_mp_bci),
        # *BatistaContextInfo.build_from_dir('./data/marino_batista/earl_multi_posture_dco_reaching', task=ExperimentalTask.marino_batista_mp_reaching),
        # *BatistaContextInfo.build_from_dir('./data/marino_batista/nigel_multi_posture_bci', task=ExperimentalTask.marino_batista_mp_bci),
        # *BatistaContextInfo.build_from_dir('./data/marino_batista/nigel_multi_posture_dco_reaching', task=ExperimentalTask.marino_batista_mp_reaching),
        # *BatistaContextInfo.build_from_dir('./data/marino_batista/rocky_multi_posture_bci', task=ExperimentalTask.marino_batista_mp_bci),
        # *BatistaContextInfo.build_from_dir('./data/marino_batista/rocky_multi_posture_dco_reaching', task=ExperimentalTask.marino_batista_mp_reaching),
    ])
