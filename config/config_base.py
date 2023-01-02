from typing import List, Optional, Union, Any, Tuple, Dict
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from omegaconf import MISSING

LENGTH = 'length'

# Convention note to self - switching to lowercase, which is more readable and much less risky now that
# config is typed
class Architecture(Enum):
    ndt = 'ndt'

class ModelTask(Enum):
    icms_one_step_ahead = 'icms_one_step_ahead'
    infill = 'infill'

    # Time-varying - these tasks are currently implemented by matching time-varying input shape
    # But could hypothetically call for enc-dec etc
    heldout_decoding = 'heldout_decoding'
    kinematic_decoding = 'kinematic_decoding'

    # Trial-summarizing
    detection_decoding = 'detection_decoding'

class Metric(Enum):
    # Monitoring metrics to log. Losses are automatically included in lgos.
    bps = 'bps'
    co_bps = 'co-bps'
    block_co_bps = 'block-co-bps'
    kinematic_r2 = 'kinematic_r2'
    all_loss = 'all_loss'

class Output(Enum):
    # Various keys for different vectors model produces
    logrates = 'logrates' # unnormalized
    heldout_logrates = 'heldout_logrates'
    rates = 'rates'
    heldout_rates = 'heldout_rates'
    poisson_loss = 'poisson_loss'
    features = 'features'
    spikes = 'spikes' # for debugging

@dataclass
class TaskConfig:
    r"""
        These are _model_ tasks, not experimental tasks.
        Beginning experiments will be pretrain -> fine-tune, but we will try to make migrating to multi-task easy.
    """
    # TODO support multitask tuning (rather, think of a scenario where that would be needed?)
    tasks: List[ModelTask] = field(default_factory=lambda: [ModelTask.infill])

    # infill
    mask_ratio: float = 0.25 # we don't have any schedule right now - the smaller this is, the higher the ceiling (probably), the slower the training
    mask_token_ratio: float = 0.8
    mask_random_ratio: float = 0.2 # It's really important to keep this quite high (in fact, anything lower than full seems to break)
    mask_random_shuffle: bool = False # doesn't actually seem more helpful

    metrics: List[Metric] = field(default_factory=lambda: [Metric.bps])
    outputs: List[Output] = field(default_factory=lambda: [])

    freeze_backbone: bool = False

    linear_head: bool = False

@dataclass
class TransformerConfig:
    n_state: int = 256
    n_heads: int = 4
    n_layers: int = 6
    feedforward_factor: float = 1.
    dropout: float = 0.2 # applies generically
    activation: str = 'gelu'
    pre_norm: bool = False
    # causal: bool = True # Pretty sure this should be passed in by task, not configured

    # Optional pattern for phasing in config?
    # fixup_init: Optional[bool] = False # doesn't seem useful

    # Position
    learnable_position: bool = False
    scale_sin: bool = False # per https://proceedings.mlr.press/v162/hua22a/hua22a.pdf

    max_trial_length: int = 1500 # Ideally we can bind this to DatasetConfig.max_trial_length

class EmbedStrat(Enum):
    # Embedding strategies
    none = "" # Just ignore context
    token = 'token' # Embed context as a token
    token_add = 'token_add' # Like token, but gets added instead of being context. Typically used for array embed, because it differentiates within trial.
    concat = 'concat' # concat embedding and downproject

    # raw data input specific
    project = 'project'
    unique_project = 'unique_project' # learn a separate projection per context
    mirror_project = 'mirror_project'

@dataclass
class ModelConfig:
    hidden_size: int = 256 # For parts outside of backbones
    arch: str = Architecture.ndt
    transformer: TransformerConfig = TransformerConfig()

    half_precision: bool = True
    lr_init: float = 0.0005 # be careful of interxn with bsz
    lr_schedule: str = 'cosine_warmup'
    # one of 'fixed' (default), 'cosine_warmup', 'linear_warmup'
    lr_ramp_init_factor: float = 0.1
    lr_ramp_steps: int = 50 # epochs # targeting ~10k steps, so this highly depends on bsz/batches per epoch. If we're under 100K items though, 50 is a lower bound.
    lr_decay_steps: int = 1000 # epochs (for cosine)
    lr_min: float = 1e-6

    activation: str = 'gelu' # gelu

    weight_decay: float = 0.01
    dropout: float = 0.2 # not inherited by transformer (typically just for model IO)
    # The objective. Not intended to be multitask right now; intent is pretrain/fine-tune.
    task: TaskConfig = TaskConfig()

    # Speed the learning rates of parameters that are freshly initialized (intended for fine-tuning)
    accelerate_new_params: float = 1.0

    # Spike prediction tasks
    lograte: bool = True

    # A few possible strategies for incorporating context information
    # "token" (this is the simplest and thus ideal one)
    # "add" (add representations)
    # "project" (have a context-specific read-in layer)
    # "" - ignore context

    # Trial level
    session_embed_strategy: EmbedStrat = EmbedStrat.token
    session_embed_size: int = 256 # TODO can we bind this?
    subject_embed_strategy: EmbedStrat = EmbedStrat.none # TODO update this once we consider mixed batches
    subject_embed_size: int = 256 # TODO can we bind this?
    task_embed_strategy: EmbedStrat = EmbedStrat.none # * we're not planning on going multitask in near future, so please hold.

    # This needs a separate API from the rest, likely, tied to readin.
    array_embed_strategy: EmbedStrat = EmbedStrat.none # ? maybe subsumed by subject
    array_embed_size: int = 256 # TODO can we bind this?

    # Closely related to, but not quite, array embed strategy.
    # Array embed strategy describes how we should provide information about array
    # Readin strategy describes IO.
    # Only when readin strategy is `token` does array embed get used.
    readin_strategy: EmbedStrat = EmbedStrat.token
    readin_dim: int = 32 # think of this as "PCs" or whatever
    readout_strategy: EmbedStrat = EmbedStrat.none

    # Timestep level
    # "concat" becomes a valid strategy at this point
    stim_embed_strategy: EmbedStrat = EmbedStrat.token
    heldout_neuron_embed_strategy: EmbedStrat = EmbedStrat.token # Not even sure if there's a different way here.
    # There should maybe be a section for augmentation/ablation, but that is low pri.

class DataKey(Enum):
    # TODO need more thinking about this. Data is heterogenuous, can we maintain a single interface
    # What is the right we to specify we want some type of array?
    spikes = 'spikes'
    stim = 'stim' # icms
    heldout_spikes = 'heldout_spikes' # for co-bps

    bhvr_vel = 'bhvr_vel'

class MetaKey(Enum):
    r"""
        Keys that are (potentially) tracked in `meta_df`
    """
    trial = 'trial'
    session = 'session'
    subject = 'subject'
    array = 'array'
    task = 'task'
    unique = 'unique' # default unique identifier

    # Note these two are trial-wise metadata, and are stored in meta.csv. Currently easier to just store string 'split' and 'path' rather than parse out the enums from the csv.
    split = 'split' # for NLB, sometimes data is loaded that has special labels/should be processed differently
    path = 'path'

@dataclass
class ExperimentalConfig:
    r"""
        It seems plausible we'll want to specify the arrays to use from different datasets with some granularity.
        For example, some stim experiments really only have good sensory array data or motor array data.
        For now, we will specify this at the level of experimental task. Note though, that we need to additionally specify selection per subject.

        I will use a somewhat heavyhanded strategy for now
        - Each dataset/subject only provides some arrays (which have subject-specific hashes)
        - Configured task arrays are keys that indicate which of these arrays should be used
        - It is assumed that subjects will not have all of these - some of these arrays belong to other subjects
        - For now we will require all full explicit array names to be specified

        Additionally, we would like to be able to specify when to group arrays together or not.
        - Probably the strategy for doing this will be array group aliases
            - This alias must propagate in both meta info and data - data should be stored per meta info.
        - It may be advantageous, or may not be advantageous, to split or group arrays.
        - More tokens, especially for distant arrays, is likely useful. However, memory is quadratic in tokens.
        * TODO Think more about this
    """
    arrays: List[str] = field(default_factory=lambda: []) # Empty list means don't filter

    def reproc_dict(self) -> Dict[str, List[str]]:
        r"""
            Dictionary of attrs that should trigger a reprocessing events
        """
        return {}

@dataclass
class RTTConfig(ExperimentalConfig):
    chop_size_ms: int = 1000
    load_covariates: bool = True

    def reproc_dict(self):
        return {'chop_size_ms': self.chop_size_ms}

@dataclass
class MazeConfig(ExperimentalConfig):
    chop_size_ms: int = 0 # no chop
    load_covariates: bool = False
    pretrial_time_s: float = 0.25
    posttrial_time_s: float = 0.1
    max_length_ms: int = 1000

    def reproc_dict(self):
        return {
            'chop_size_ms': self.chop_size_ms,
            'max_length_ms': self.max_length_ms,
            'pretrial_time_s': self.pretrial_time_s,
            'posttrial_time_s': self.posttrial_time_s,
        }

@dataclass
class NLBConfig(ExperimentalConfig):
    heldout_neurons: int = 32 # for RTT

@dataclass
class DatasetConfig:
    root_dir: Path = Path("./data")
    preprocess_suffix: str = 'preprocessed'

    dataset_seed: int = 0 # for shuffling/splitting etc
    r"""
        Specifies the source datasets.
        - datasets accepts lists of strings that point to registered data files; this pointer can be one of:
            - paths to data files themselves
            - aliases (from registration)
            - lightweight regex for _aliases_ (not paths). Note this is regex, not glob.
    """
    datasets: List[str] = field(default_factory=lambda: [])
    r"""
        `data_keys` and `meta_keys` specify the attributes of the dataset are served.
    """
    data_keys: List[DataKey] = field(
        default_factory=lambda: [DataKey.spikes]
    )
    meta_keys: List[MetaKey] = field(
        default_factory=lambda: [MetaKey.unique, MetaKey.session, MetaKey.array]
    ) # JY recommends providing array meta info, but thinks the system should be designed to not error without.

    split_key: MetaKey = MetaKey.unique
    # ==== Data parsing/processing ====
    bin_size_ms: int = 2
    pad_batches: bool = True # else, trim batches to the shortest trial
    max_trial_length: int = 1500 # in bins

    # Pad to this number of channels per array group
    # If set to 0, will skip padding checks.
    max_channels: int = 0 # ! TODO add smart inference (take max over array reports)

    # Pad to this number of arrays (for meta and data alike). Must be >= 1
    max_arrays: int = 1

    # Experimental Task configuration - matching registered names
    # Note - we choose to put task specific things here rather than ModelConfig as model will read the relevant variables
    # from `data_attrs`. Tasks may be specified to e.g. load specific subsets of targets rather than full data
    # and so Dataset must know about this; and probably better to propagate this to ModelConfig than to have
    # to track it in two places.
    passive_icms: ExperimentalConfig = ExperimentalConfig()
    nlb_maze: NLBConfig = NLBConfig()
    nlb_rtt: NLBConfig = NLBConfig()
    churchland_maze: MazeConfig = MazeConfig()
    odoherty_rtt: RTTConfig = RTTConfig()

@dataclass
class TrainConfig:
    epochs: int = 1000
    steps: int = 0 # Prefer to specify steps over epochs for FLOP consistency (pretty loose), but most other training settings are on epochs
    batch_size: int = 64
    patience: int = 25 # these are in units of val checks (epochs)
    log_grad: bool = False
    gradient_clip_val: float = 1.0
    accumulate_batches: int = 1
    profiler: str = ""
    val_check_interval: int = 100 # these are in steps, but mostly isn't used # TODO deprecate

@dataclass
class RootConfig:
    seed: int = 0
    tag: str = "" # i.e. experiment variant, now an optional tag (since hydra consumes file, we can't use the filename for experiment name. Specify if you want.)
    experiment_set: str = ""
    default_root_dir: Path = Path("./data/runs").resolve()
    wandb_project: str = "context_general_bci"
    wandb_api_key_path: Path = Path("/home/joelye/.wandb_api").resolve()
    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = DatasetConfig()
    train: TrainConfig = TrainConfig()

    # wandb ids
    init_from_id: str = "" # for initializing weights
    init_tag: str = "bps"
    load_from_id: str = "" # for resuming training. takes precedent over init_from_id

    exp: Any = MISSING # delta config, provide via yaml and on CLI as `+exp=<test>.yaml`