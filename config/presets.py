from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

r"""
    In Hydra, experimental presets can be declared either in YAML, or via the ConfigStore API.
    We will use ConfigStore API for the type safety.
"""

from .config_base import *

cs = ConfigStore.instance()

@dataclass
class InfillTaskConfig(TaskConfig):
    task: ModelTask = ModelTask.infill


@dataclass
class PretrainingModelConfig(ModelConfig):
    r"""
        BERT does 10K ramp, 1M full. We are ~2 orders of magnitude data smaller.
        Our "steps" below are in epochs - so adjust appropriately. Currently
        ~10 batches per epoch in 2K trials (unaggregate)
    """
    task: TaskConfig = field(default_factory=InfillTaskConfig)
    lr_ramp_steps: int = 500
    lr_decay_steps: int = 10000
cs.store(group="model", name="pretrain", node=PretrainingModelConfig)

@dataclass
class PretrainingSmallModelConfig(ModelConfig):
    task: TaskConfig = field(default_factory=InfillTaskConfig)
    lr_ramp_steps: int = 3000
    lr_decay_steps: int = 100000
cs.store(group="model", name="pretrain_small", node=PretrainingSmallModelConfig)

@dataclass
class FinetuningModelConfig(ModelConfig):
    lr_init: float = 2e-5
    lr_ramp_steps: int = 2000 # epochs tend to be small
    lr_decay_steps: int = 10000 # epochs tend to be small

cs.store(group="model", name="finetune", node=FinetuningModelConfig)

@dataclass
class NLBModelTaskConfig(TaskConfig):
    task: ModelTask = ModelTask.heldout_decoding
    metrics: List[Metric] = field(default_factory=lambda: [Metric.co_bps, Metric.block_co_bps])
    outputs: List[Output] = field(default_factory=lambda: [Output.heldout_logrates])

cs.store(group='model/task', name='nlb', node=NLBModelTaskConfig)
@dataclass
class NLBModelConfig(ModelConfig):
    task: TaskConfig = field(default_factory=NLBModelTaskConfig)

cs.store(group="model", name="nlb", node=NLBModelConfig)

@dataclass
class PretrainConfig(TrainConfig):
    epochs: int = 10000
    batch_size: int = 256
    patience: int = 100
cs.store(group="train", name="pretrain", node=PretrainConfig)

@dataclass
class NLBTrainConfig(TrainConfig):
    epochs: int = 50000 # epochs tend to be small
    batch_size: int = 128
    patience: int = 500

cs.store(group="train", name="nlb", node=NLBTrainConfig)
cs.store(group="train", name="small", node=NLBTrainConfig) # alias

@dataclass
class RTTNLBDataConfig(DatasetConfig):
    r"""
        Default configuration for RTT NLB fine-tuning
    """
    bin_size_ms: int = 5
    datasets: List[str] = field(default_factory=lambda: ['mc_rtt'])
    max_channels: int = 98
    data_keys: List[DataKey] = field(default_factory=lambda: [DataKey.spikes, DataKey.heldout_spikes])

cs.store(group="dataset", name="rtt_nlb", node=RTTNLBDataConfig)

@dataclass
class MCMazeExpConfig(NLBConfig):
    heldout_neurons: int = 45

cs.store(group='dataset/nlb_maze', name='mc_maze', node=MCMazeExpConfig)

@dataclass
class MazeNLBDataConfig(DatasetConfig):
    r"""
        Default configuration for all maze datasets NLB fine-tuning
    """
    bin_size_ms: int = 5
    datasets: List[str] = field(default_factory=lambda: ['mc_maze$'])
    max_channels: int = 137
    data_keys: List[DataKey] = field(default_factory=lambda: [DataKey.spikes, DataKey.heldout_spikes])
    nlb_maze: NLBConfig = field(default_factory=MCMazeExpConfig)

cs.store(group="dataset", name="maze_nlb", node=MazeNLBDataConfig)

@dataclass
class RTTExpConfig(NLBConfig):
    heldout_neurons: int = 32

@dataclass
class ODohertyExpConfig(RTTConfig):
    arrays: List[str] = field(default_factory=lambda: ['Indy-M1', 'Loco-M1'])

cs.store(group='dataset/nlb_rtt', name='mc_rtt', node=RTTExpConfig)

@dataclass
class RTTNLBDataConfig(DatasetConfig):
    r"""
        Default configuration for all maze datasets NLB fine-tuning
    """
    bin_size_ms: int = 5
    datasets: List[str] = field(default_factory=lambda: ['mc_rtt', 'odoherty_rtt.*'])
    max_channels: int = 98
    max_arrays: int = 1
    data_keys: List[DataKey] = field(default_factory=lambda: [DataKey.spikes, DataKey.heldout_spikes])
    nlb_rtt: NLBConfig = field(default_factory=RTTExpConfig)
    odoherty_rtt: ODohertyExpConfig = field(default_factory=ODohertyExpConfig)

cs.store(group="dataset", name="rtt_nlb", node=RTTNLBDataConfig)
