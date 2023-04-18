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
    tasks: List[ModelTask] = field(default_factory=lambda: [ModelTask.infill])

@dataclass
class SmallTransformerConfig(TransformerConfig):
    n_state: int = 128
    dropout: float = 0.6

@dataclass
class SmallTransformerConfigLessDrop(TransformerConfig):
    n_state: int = 128
    dropout: float = 0.3


@dataclass
class PretrainingModelConfig(ModelConfig):
    r"""
        BERT does 10K ramp, 1M full. We are ~2 orders of magnitude data smaller.
        Our "steps" below are in epochs - so adjust appropriately. Currently
        ~10 batches per epoch in 2K trials (unaggregate)
    """
    task: TaskConfig = field(default_factory=InfillTaskConfig)
    lr_ramp_steps: int = 500
    lr_decay_steps: int = 10000 # this is not that much for small models
    dropout: float = 0.1
    hidden_size: int = 128
    transformer: TransformerConfig = field(default_factory=SmallTransformerConfigLessDrop)
cs.store(group="model", name="pretrain", node=PretrainingModelConfig)

@dataclass
class BaseTransformerConfig(TransformerConfig):
    n_state: int = 256
    n_heads: int = 4
    n_layers: int = 6
    dropout: float = 0.1
    learnable_position: bool = True
    max_trial_length: int = 250

@dataclass
class PretrainingNewModelConfig(ModelConfig):
    # A little more informed after initial experimentation
    task: TaskConfig = field(default_factory=InfillTaskConfig)
    lr_ramp_steps: int = 100
    lr_decay_steps: int = 2500
    dropout: float = 0.1
    hidden_size: int = 256

    transformer: TransformerConfig = field(default_factory=BaseTransformerConfig)

    # base config: 6 layers, 256 hidden, 4 heads
cs.store(group="model", name="pretrain_2x", node=PretrainingNewModelConfig)

@dataclass
class PretrainingSmallModelConfig(ModelConfig):
    task: TaskConfig = field(default_factory=InfillTaskConfig)
    lr_ramp_steps: int = 3000
    lr_decay_steps: int = 100000
    dropout: float = 0.6
    hidden_size: int = 128
    session_embed_size: int = 128
    subject_embed_size: int = 128
    array_embed_size: int = 128
    transformer: TransformerConfig = field(default_factory=SmallTransformerConfig)

cs.store(group="model", name="pretrain_small", node=PretrainingSmallModelConfig)

@dataclass
class FinetuningModelConfig(ModelConfig):
    lr_init: float = 5e-5
    lr_ramp_steps: int = 3000 # epochs tend to be small
    lr_decay_steps: int = 100000 # epochs tend to be small

cs.store(group="model", name="finetune", node=FinetuningModelConfig)

@dataclass
class AcceleratedTuning(ModelConfig):
    lr_init: float = 5e-5
    lr_ramp_steps: int = 1000
    lr_decay_steps: int = 10000
    accelerate_new_params: float = 10.0

cs.store(group="model", name="accel_tune", node=AcceleratedTuning)

@dataclass
class FlatEncDecTransformerConfig(TransformerConfig):
    n_state: int = 256
    n_heads: int = 4
    n_layers: int = 6
    dropout: float = 0.1
    flat_encoder: bool = True
    learnable_position: bool = True
    max_trial_length: int = 250

@dataclass
class FlatEncDecTaskConfig(TaskConfig):
    tasks: List[ModelTask] = field(default_factory=lambda: [ModelTask.shuffle_infill])
    metrics: List[Metric] = field(default_factory=lambda: [])

@dataclass
class FlatEncDecModelConfig(ModelConfig):
    lr_ramp_steps: int = 100
    lr_decay_steps: int = 2500 # we update to be even more conservative with decay, we just want to prevent killing too soon for scientific investigations
    # lr_decay_steps: int = 1000
    dropout: float = 0.1
    hidden_size: int = 256
    encode_decode: bool = True
    transform_space: bool = True
    spike_embed_style: EmbedStrat = EmbedStrat.token
    transformer: TransformerConfig = field(default_factory=FlatEncDecTransformerConfig)
    task: TaskConfig = field(default_factory=FlatEncDecTaskConfig)

cs.store(group="model", name="flat_enc_dec", node=FlatEncDecModelConfig)

@dataclass
class NLBModelTaskConfig(TaskConfig):
    tasks: List[ModelTask] = field(default_factory=lambda: [ModelTask.heldout_decoding])
    metrics: List[Metric] = field(default_factory=lambda: [Metric.co_bps, Metric.block_co_bps])
    outputs: List[Output] = field(default_factory=lambda: [Output.heldout_logrates])

cs.store(group='model/task', name='nlb', node=NLBModelTaskConfig)
@dataclass
class NLBModelConfig(ModelConfig):
    task: TaskConfig = field(default_factory=NLBModelTaskConfig)

cs.store(group="model", name="nlb", node=NLBModelConfig)

@dataclass
class BhvrDecodeFlatTaskConfig(FlatEncDecTaskConfig):
    tasks: List[ModelTask] = field(default_factory=lambda: [ModelTask.kinematic_decoding])
    metrics: List[Metric] = field(default_factory=lambda: [Metric.kinematic_r2])
    decode_strategy: EmbedStrat = EmbedStrat.token
    decode_separate: bool = True

    decode_time_pool: str = 'mean'

cs.store(group='model/task', name='bhvr_decode_flat', node=BhvrDecodeFlatTaskConfig)

@dataclass
class JointBhvrDecodeFlatTaskConfig(FlatEncDecTaskConfig):
    tasks: List[ModelTask] = field(default_factory=lambda: [ModelTask.shuffle_infill, ModelTask.kinematic_decoding])
    metrics: List[Metric] = field(default_factory=lambda: [Metric.kinematic_r2])
    task_weights: List[float] = field(default_factory=lambda: [1.0, 20.0]) # so they're both on order of 0.3 (for bin size 20ms)

    decode_strategy: EmbedStrat = EmbedStrat.token
    decode_separate: bool = True

    decode_time_pool: str = 'mean'

cs.store(group='model/task', name='joint_bhvr_decode_flat', node=JointBhvrDecodeFlatTaskConfig)

@dataclass
class JointHeldoutDecodeTaskConfig(FlatEncDecTaskConfig):
    tasks: List[ModelTask] = field(default_factory=lambda: [ModelTask.shuffle_infill, ModelTask.heldout_decoding])
    metrics: List[Metric] = field(default_factory=lambda: [Metric.co_bps, Metric.block_co_bps])

    decode_strategy: EmbedStrat = EmbedStrat.token
    decode_separate: bool = False
cs.store(group='model/task', name='joint_heldout_decode', node=JointHeldoutDecodeTaskConfig)

@dataclass
class BhvrDecodeTaskConfig(InfillTaskConfig):
    tasks: List[ModelTask] = field(default_factory=lambda: [ModelTask.kinematic_decoding])
    metrics: List[Metric] = field(default_factory=lambda: [Metric.kinematic_r2])
    decode_strategy: EmbedStrat = EmbedStrat.project

    decode_time_pool: str = "mean"

cs.store(group='model/task', name='bhvr_decode', node=BhvrDecodeTaskConfig)

@dataclass
class JointBhvrDecodeTaskConfig(InfillTaskConfig):
    tasks: List[ModelTask] = field(default_factory=lambda: [ModelTask.infill, ModelTask.kinematic_decoding])
    task_weights: List[float] = field(default_factory=lambda: [1.0, 20.0]) # so they're both on order of 0.3 (for bin size 20ms)

    metrics: List[Metric] = field(default_factory=lambda: [Metric.bps, Metric.kinematic_r2])
    decode_strategy: EmbedStrat = EmbedStrat.project

cs.store(group='model/task', name='joint_bhvr_decode', node=JointBhvrDecodeTaskConfig)

@dataclass
class PretrainConfig(TrainConfig):
    epochs: int = 4000
    batch_size: int = 128
    patience: int = 50
# As we hit >10K datapts, we typically see convergence in ~800 epochs at most.
cs.store(group="train", name="pretrain", node=PretrainConfig)

@dataclass
class NLBTrainConfig(TrainConfig):
    epochs: int = 50000 # epochs tend to be small
    batch_size: int = 64
    autoscale_batch_size: bool = False
    patience: int = 2000
    # patience: int = 4000

cs.store(group="train", name="nlb", node=NLBTrainConfig)
cs.store(group="train", name="small", node=NLBTrainConfig) # alias

@dataclass
class FineTuneConfig(TrainConfig):
    epochs: int = 10000
    batch_size: int = 64 # arbitrary, expectation is autoscale
    patience: int = 200

cs.store(group="train", name="finetune", node=FineTuneConfig)

@dataclass
class BaseDataConfig(DatasetConfig):
    """
        Base configuration for all datasets.
        We tend to only use M1.
    """
    bin_size_ms: int = 20
    max_tokens: int = 8192
    max_length_ms: int = 4000 # most data is much shorter, though.
    data_keys: List[DataKey] = field(default_factory=lambda: [DataKey.spikes])
    meta_keys: List[MetaKey] = field(default_factory=lambda: [
        MetaKey.unique, MetaKey.array, MetaKey.subject, MetaKey.session, MetaKey.task
    ])
    odoherty_rtt: RTTConfig = field(default_factory=lambda: RTTConfig(
        arrays=['Indy-M1_all', 'Loco-M1_all'],
        # arrays=['Indy-M1', 'Loco-M1'],
        include_sorted=True,
    ))
    gallego_co: ExperimentalConfig = field(default_factory=lambda: ExperimentalConfig(
        arrays=['Chewie-M1', 'Mihi-M1']
    ))
    churchland_misc: ExperimentalConfig = field(default_factory=lambda: ExperimentalConfig(
        arrays=["Reggie-M1", "Nitschke-M1", "Jenkins-M1"]
    ))
    pitt_co: PittConfig = field(default_factory=lambda: PittConfig(
        arrays=["CRS02b-lateral_m1", "CRS02b-medial_m1", "CRS07-lateral_m1", "CRS07-medial_m1"]
    ))

cs.store(group="dataset", name="base", node=BaseDataConfig)

@dataclass
class FlatDataConfig(BaseDataConfig):
    serve_tokenized: bool = True
    serve_tokenized_flat: bool = True
    # Liberally set upper bound, since flat models only use this to determine position encoder capacity.
    max_arrays: int = 2
    max_channels: int = 288

cs.store(group="dataset", name="flat", node=FlatDataConfig)

@dataclass
class ODohertyUnsort(FlatDataConfig):
    odoherty_rtt: RTTConfig = field(default_factory=lambda: RTTConfig(
        # arrays=['Indy-M1_all', 'Loco-M1_all'],
        arrays=['Indy-M1', 'Loco-M1'],
        include_sorted=False,
    ))
cs.store(group="dataset", name="odoherty_unsort_flat", node=ODohertyUnsort)

@dataclass
class RTTDataConfig(DatasetConfig):
    r"""
        Default configuration for all RTT datasets
    """
    bin_size_ms: int = 20 # 20 is the new king...
    # bin_size_ms: int = 5
    datasets: List[str] = field(default_factory=lambda: ['mc_rtt', 'odoherty_rtt.*'])
    max_channels: int = 98
    data_keys: List[DataKey] = field(default_factory=lambda: [DataKey.spikes])
    max_arrays: int = 1
    meta_keys: List[MetaKey] = field(default_factory=lambda: [
        MetaKey.unique, MetaKey.array, MetaKey.subject, MetaKey.session,
    ])

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
class MazeDataConfig(DatasetConfig):
    r"""
        Default configuration for all maze datasets
    """
    bin_size_ms: int = 5
    datasets: List[str] = field(default_factory=lambda: ['mc_maze.*', 'churchland_maze_*'])
    max_channels: int = 137
    data_keys: List[DataKey] = field(default_factory=lambda: [DataKey.spikes])
    nlb_maze: NLBConfig = field(default_factory=MCMazeExpConfig)
    max_arrays: int = 2
    meta_keys: List[MetaKey] = field(default_factory=lambda: [
        MetaKey.unique, MetaKey.array, MetaKey.subject, MetaKey.session,
    ])
cs.store(group="dataset", name="maze", node=MazeDataConfig)

@dataclass
class MazeNLBDataConfig(DatasetConfig):
    r"""
        Default configuration for NLB-dataset pretrain or fine-tuning
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
class RTTDataConfig(DatasetConfig):
    r"""
        Default configuration for all maze datasets NLB fine-tuning
    """
    bin_size_ms: int = 20
    datasets: List[str] = field(default_factory=lambda: ['mc_rtt', 'odoherty_rtt.*'])
    max_channels: int = 98
    max_arrays: int = 1
    data_keys: List[DataKey] = field(default_factory=lambda: [DataKey.spikes, DataKey.heldout_spikes])
    nlb_rtt: NLBConfig = field(default_factory=RTTExpConfig)
    odoherty_rtt: ODohertyExpConfig = field(default_factory=ODohertyExpConfig)

cs.store(group="dataset", name="rtt", node=RTTDataConfig)

r"""
    Some experiment specific presets
"""
@dataclass
class SingleSessionTrainConfigExp1(TrainConfig):
    patience: int = 250
    autoscale_batch_size: bool = False
    batch_size: int = 64

cs.store(group="train", name="single_session_exp1", node=SingleSessionTrainConfigExp1)

@dataclass
class Trial100TuneConfigExp2(TrainConfig):
    patience: int = 150
    autoscale_batch_size: bool = False
    batch_size: int = 32

cs.store(group="train", name="trial100_tune_exp2", node=Trial100TuneConfigExp2)
