from typing import List, Optional, Union, Any, Tuple, Dict
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from omegaconf import MISSING

DEFAULT_KIN_LABELS = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gx', 'gy', 'f', 'null'] # Null dimension only used for padding in tokenized case
REACH_DEFAULT_KIN_LABELS = ['y', 'z']
REACH_DEFAULT_3D_KIN_LABELS = ['x', 'y', 'z']
EMG_CANON_LABELS = ['EMG_FCU', 'EMG_EDCr', 'EMG_ECU', 'EMG_ECRb', 'EMG_ECRl', 'EMG_FDP', 'EMG_FCR'] # Just an order pulled from xds tutorial: https://github.com/limblab/adversarial_BCI/blob/main/xds_tutorial.ipynb.
# ! Not actually really reliable - these are canon for Jango iso, another set of labels are used for Greyson

LENGTH = 'length'

# Convention note to self - switching to lowercase, which is more readable and much less risky now that
# config is typed
class Architecture(Enum):
    ndt = 'ndt'
    flash_ndt = 'flash_ndt'

class ModelTask(Enum):
    next_step_prediction = 'next_step' # Decoder-only path, global modality
    infill = 'infill'

    return_context = 'return_context'
    return_infill = 'return_infill'

    spike_context = 'spike_context'
    shuffle_next_step_prediction = 'shuffle_next_step_prediction'
    shuffle_infill = 'shuffle_infill'
    spike_infill = 'spike_infill'

    # Time-varying - these tasks are currently implemented by matching time-varying input shape
    # But could hypothetically call for enc-dec etc
    heldout_decoding = 'heldout_decoding'
    kinematic_decoding = 'kinematic_decoding'
    kinematic_classification = 'kinematic_classification'
    kinematic_context = 'kinematic_context'
    kinematic_infill = 'kinematic_infill'

    # Trial-summarizing
    detection_decoding = 'detection_decoding'

    constraints = 'constraints'


class Metric(Enum):
    # Monitoring metrics to log. Losses are automatically included in lgos.
    bps = 'bps'
    co_bps = 'co-bps'
    block_co_bps = 'block-co-bps'
    kinematic_r2 = 'kinematic_r2'
    kinematic_r2_thresh = 'kinematic_r2_thresh' # a clone that will threshold out extremely low velocities to match Pitt settings
    kinematic_acc = 'kinematic_acc'
    kinematic_mse = 'kinematic_mse'
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

    behavior = 'behavior'
    behavior_pred = 'behavior_pred' # pred, not logits (in classification case)
    behavior_logits = 'behavior_logits' # logits, not pred (in classification case)
    behavior_query_mask = 'behavior_query_mask' # Which ones were actual predictions vs inputs?

    # Debug
    pooled_features = 'pooled_features'

class DataKey(Enum):
    # DataKey are time-varying and typically served with spikes
    spikes = 'spikes'
    stim = 'stim' # icms
    heldout_spikes = 'heldout_spikes' # for co-bps

    bhvr_vel = 'bhvr_vel' # general continuous covariate key
    bhvr_acc = 'bhvr_acc'
    bhvr_force = 'bhvr_force'


    covariate_time = 'covariate_time'
    covariate_space = 'covariate_space'
    covariate_labels = 'covariate_labels' # For annotating sparse bhvr

    # Assist (for BCI exps)
    # Note these are timevarying because control toggles on and off often in historical BCI data (e.g. in trialized exps).
    constraint = 'constraints' # triplet of active, passive, brain control
    # active_assist = 'active_assist' # Autopilot (e.g. observation). Should be 1 at test.
    # passive_assist = 'passive_assist' # Constraint based (e.g. ortho). Should be 0 at test.
    # brain_control = 'brain_control' # Extent to which the neural data is driving behavior. Should be 1-active assist during task phases.
    constraint_time = 'constraints_time' # for sparse constraints
    constraint_space = 'constraints_space' # TODO unify single/plural in key/value here (should trigger reproc)

    # Inclusion of return will auto-include reward. Note that return changepoints are strict superset of reward changepoints, as return changepoints include future reward showing up in horizon as well as reward toggle in present timepoint.
    task_return = 'task_return' # Reward conditioned behavior cloning
    task_reward = 'task_reward' # Reward conditioned behavior cloning
    task_return_time = 'task_return_time'

    time = 'time'
    position = 'position' # space, however you want to think about it. Tracks channel cluster.
    padding = 'padding'
    extra = 'extra' # utility for decoding
    extra_time = 'extra_time'
    extra_position = 'extra_position'

class MetaKey(Enum):
    r"""
        Keys that are (potentially) tracked in `meta_df`; should be trial level metadata.
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


class EmbedStrat(Enum):
    # Embedding strategies, used in several contexts (overloaded)
    none = "" # Just ignore context
    token = 'token' # Embed context as a token
    token_add = 'token_add' # Like token, but gets added instead of being context. Typically used for array embed, because it differentiates within trial.
    concat = 'concat' # concat embedding and downproject

    # readin specific
    project = 'project'
    unique_project = 'unique_project' # learn a separate projection per context
    mirror_project = 'mirror_project'

    readin_cross_attn = 'cross_attn'
    contextual_mlp = 'contextual_mlp' # feed raw context.

@dataclass
class TaskConfig:
    r"""
        These are _model_ tasks, not experimental tasks.
        Beginning experiments will be pretrain -> fine-tune, but we will try to make migrating to multi-task easy.
    """
    tasks: List[ModelTask] = field(default_factory=lambda: [ModelTask.infill])
    task_weights: List[float] = field(default_factory=lambda: [1., 1.])
    # List of session IDs to ignore supervised loss for. Using for mc_rtt pilot
    blacklist_session_supervision: List[str] = field(default_factory=lambda: [])

    # Alignment can be done either with an adversarial loss (not really made working...) or KL on the multivariate KL.
    # adversarial_classify_lambda: float = 0.0
    # kl_lambda: float = 0.0
    # alignment_distribution_path: str = ""

    # infill
    mask_ratio: float = 0.25 # we don't have any schedule right now - the smaller this is, the higher the ceiling (probably), the slower the training
    context_prompt_time_thresh: int = 0 # Supporting in-context learning by providing minimal start of sequence
    context_prompt_time_thresh_min: int = 0
    # Based on timestep of tokens (in token bin units)
    # For autoregressive models, this just means we start evaluating loss after N tokens (and is probably honestly unnecessary)
    prefix_ratio: float = 0.0 # ratio of using prefix loss - i.e. only count loss on maskout suffix. Assumes negative context_prompt_time_thresh
    # For shuffle based non-autoregressive models, this means never shuffle out the first N tokens during decoding, we assume those are provided.

    # These ratios are only relevant for non-asymmetric path (i.e. defunct)
    mask_token_ratio: float = 0.8
    mask_random_ratio: float = 0.2 # It's really important to keep this quite high (in fact, anything lower than full seems to break)
    mask_random_shuffle: bool = False # doesn't actually seem more helpful

    spike_loss: str = 'poisson' # poisson or cross_entropy
    cross_ent_soften: bool = True

    metrics: List[Metric] = field(default_factory=lambda: [Metric.bps])
    outputs: List[Output] = field(default_factory=lambda: [])

    freeze_backbone: bool = False
    freeze_embed: bool = False
    freeze_all: bool = False # stricter than above, only allows embedding

    linear_head: bool = False
    unique_no_head: bool = False # overrides above

    # kinematic decode
    covariate_mask_ratio: float = 1.0 # If < 1.0, unmask some covariates and send them to encoder. Assumes asymmetric path
    # * Major flag for NDT3

    behavior_lag: int = 0 # in ms
    behavior_target: DataKey = DataKey.bhvr_vel
    behavior_lag_lookahead: bool = True # if true, allow lookahead up to `lag`. Only applied in causal path
    behavior_fit_thresh: float = 0.0 # exclude from loss, timesteps with values (velocities) less than this
    behavior_metric_thresh: float = 0.0001 # exclude from r2, timesteps with values (velocities) less than this
    covariate_blacklist_dims: List[int] = field(default_factory=lambda: []) # list of dims to exclude from covariate decoding (for regression testing)
    encode_constraints: bool = False # Add constraints if available, currently implemented in covariate path
    use_constraint_cls: bool = True

    # Trying to deal with incredibly noisy behavioral labels from human observation
    # By making supervision less prescriptive - expecting to reduce overfit
    behavior_contrastive: str = "" # str specifies integration style, e.g. direct sum (simpler) or e.g. rnn, use contrastive loss instead of MSE

    behavior_tolerance: float = 0.0 # if > 0, use this as a tolerance for behavior labels. If the difference between the predicted and actual behavior is less than this, don't penalize it.
    behavior_tolerance_ceil: float = 0.0 # if > 0, use this as a tolerance for behavior labels. If the difference between the predicted and actual behavior is less than this, don't penalize it.

    decode_separate: bool = False # for bhvr decoding, use a separate transformer decoder? (Only compat with EmbedStrat.token)
    decode_time_pool: str = "" # none or 'mean'
    decode_strategy: EmbedStrat = EmbedStrat.project # or EmbedStrat.token
    decode_tokenize_dims: bool = False # If true, each decode dimension gets its own token
    decode_normalizer: str = '' # If provided, use this path to normalize
    decode_quantize_classes: int = 128 # not enough... # TODO update to 256
    decode_use_shuffle_backbone: bool = False # Don't discard shuffle infill decode, take full rates as input to backbone features (useful specifically for parity on HeldoutPrediction)
    decode_label_smooth: float = 0.0 # If > 0, use this as a label smoothing factor for classifier decoding
    decode_symlog: bool = False # symlog for classification

    # Held-out neuron prediction - for integration into `ShuffleInfill` (rather than separate task)
    query_heldout: int = 0 # number of heldout neurons to query
    detach_decode_context: bool = False # reduce gradients from decoding tasks to context

@dataclass
class TransformerConfig:
    n_state: int = 256
    n_heads: int = 4
    n_layers: int = 6
    feedforward_factor: float = 1.
    dropout: float = 0.2 # applies generically
    activation: str = 'gelu'
    pre_norm: bool = False
    final_norm: bool = True # if pre-norm, add another layer norm at the end of the transformer, per Kaiming's MAE for vision and GPT
    # causal: bool = True # Pretty sure this should be passed in by task, not configured

    # Optional pattern for phasing in config?
    # fixup_init: Optional[bool] = False # doesn't seem useful

    use_biases: bool = True # TODO implement false path - remove linear and layernorm biases, efficiency
    use_attn_biases: bool = True
    initializer_range: float = 0.02 # for linear layers
    learnable_norm: bool = True # LN elementwise affine

    # Position
    learnable_position: bool = False
    rotary_position: bool = False
    scale_sin: bool = False # per https://proceedings.mlr.press/v162/hua22a/hua22a.pdf

    max_trial_length: int = 250 # This is in BINS for the position encoding, not bound to dataset config for easy transfer

    transform_space: bool = False # match ModelConfig.transform_space
    flat_encoder: bool = False # for serve_tokens_flat
    embed_space: bool = True
    max_spatial_tokens: int = 0 # 0 means infer; which is max_channels * max_arrays / chunk_size

    factorized_space_time: bool = False # will split layers evenly in space and time

    debug_force_nonlearned_position: bool = False
    debug_override_dropout_io: bool = False

    context_integration: str = "in_context" # in_context, cross_attn, or adaptive_norm (see https://arxiv.org/pdf/2212.09748.pdf)

@dataclass
class EvalConfig:
    temperature: float = 0. # For sampling. 0. is argmax, higher is more uniform
    teacher_timesteps: int = 25 # provide true labels up to N _timesteps_ in. In units of timebins
    # Specifically re: off by 1 - do we use the predictions from >= this timestep as student labels?
    use_student: bool = False # Use student predictions at next step, else drop. (For debugging constant predictions/train time parity)
    maskout_last_n: int = 0 # Assumes student path. Will allow student to fill in only if n timesteps older than present step.
    student_prob: float = 1. # If < 1, use this as a probability of using student predictions at next step, else drop. (For debugging constant predictions/train time parity)
    limit_timesteps: int = 0 # limit eval to N timesteps. In units of timebins
    student_gap: int = 0 # Timesteps since teacher to start counting predictions. Related but exclusive from maskout_last_n


@dataclass
class ModelConfig:
    compile: bool = False # use torch.compile

    hidden_size: int = 256 # For parts outside of backbones
    arch: Architecture = Architecture.ndt
    transformer: TransformerConfig = field(default_factory=lambda: TransformerConfig())

    # Asymmetric
    encode_decode: bool = False # If true, split model into encode-decode pathways per Kaiming's scaling vision/video papers.
    # This is a master flag, and changes a few pathways
    decoder_layers: int = 2
    decoder_context_integration: str = "in_context" # only implemented for behavior atm
    spike_context_integration: str = "in_context" # TODO merge into above, just testing for memory right now
    use_full_encode: bool = False # ! Major change, return all tokens in decode stream
    cm3leon_init: bool = False # Copy truncated normal params used in cm3leon https://scontent.fagc3-1.fna.fbcdn.net/v/t39.2365-6/358725877_789390529544546_1176484804732743296_n.pdf?_nc_cat=108&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=uzIGsR3Sm-QAX9dir0m&_nc_ht=scontent.fagc3-1.fna&oh=00_AfDTfWg1ZiNMx_GtFdvmQNx8gRoLjlP3lgnp2PngsUC4nQ&oe=651C2FB2


    next_step_prediction: bool = False # Major change, autoregressive path, limited compatibility with most NDT2 settigs
    # Behavioral data is nearly Markovian, nearly constant; we want to learn longer order dependencies, so upweight that learning
    # By occassionally blanking timesteps
    token_maskout: float = 0. # If true, blank the previous timestep in the backbone stream
    kinematic_token_maskout: float = 0. # Blank kinematic inputs specifically. DOUBLES AS SCHEDULE END

    fit_to_max_length: int = 0 # set to some high value for static shape, for compilation
    # e.g. max_spatial_position * max_trial_length = 16 * 250 = 4096 (amoritzing out many constraint dims)

    # Overrides above
    kinematic_token_maskout_schedule: str = "constant"
    # Schedule tracks lr schedule.
    # Constant is no schedule, constant maskout.
    # Cosine as default inspired by MaskGIT (but I made up the curriculum part, MaskGIT actually uses random).
    # Random - no schedule, sample a random value from start to end.
    kinematic_token_maskout_start: float = 0.9 # generalist still needs something, this should be less than 1
    # kinematic_token_maskout_end: float = 0.

    max_spatial_position: int = 32 # For next step prediction

    half_precision: bool = True
    # full_half_precision: bool = False # if true, use half precision for all model parameters, not just mixed precision
    lr_init: float = 0.0005 # be careful of interxn with bsz
    # lr_schedule: str = 'cosine_timm' # Preferred for stateless, rollback-able nature
    lr_schedule: str = 'cosine_warmup'
    # one of 'fixed' (default), 'cosine_warmup', 'linear_warmup'
    lr_ramp_init_factor: float = 0.1
    lr_ramp_steps: int = 50 # epochs # targeting ~10k steps, so this highly depends on bsz/batches per epoch. If we're under 100K items though, 50 is a lower bound.
    lr_interval: str = 'epoch' # 'step' or 'epoch'
    lr_decay_steps: int = 1000 # epochs (for cosine)
    lr_min: float = 1e-6

    lr_schedule_hotfix_epoch: int = 0 # If > 0, reload schedule at this epoch. For hotfixing old non-timm schedules that are stateful with new timm schedules that are stateless and directly read schedule from epoch. For rollback
    lr_schedule_hotfix_factor: float = 0.8

    activation: str = 'gelu' # gelu

    weight_decay: float = 0.01
    dropout: float = 0.2 # not inherited by transformer (typically just for model IO)
    # The objective. Not intended to be multitask right now; intent is pretrain/fine-tune.
    task: TaskConfig = field(default_factory=lambda: TaskConfig())

    # Speed the learning rates of parameters that are freshly initialized (intended for fine-tuning)
    accelerate_new_params: float = 1.0
    tune_decay: float = 0.0 # if > 0; employ decay on the learning rate of the fine-tuned parameters per layer

    # Spike prediction tasks
    lograte: bool = True

    # A few possible strategies for incorporating context information
    # "token" (this is the simplest and thus ideal one)
    # "add" (add representations)
    # "project" (have a context-specific read-in layer)
    # "" - ignore context

    init_flags: bool = True

    # Trial level
    session_embed_strategy: EmbedStrat = EmbedStrat.token
    session_embed_size: int = 256 # Bound in `propagate_config`
    session_embed_token_count: int = 1 # we'd like to increase custom capacity

    subject_embed_strategy: EmbedStrat = EmbedStrat.none
    subject_embed_size: int = 256 # Bound in `propagate_config`
    subject_embed_token_count: int = 1

    task_embed_strategy: EmbedStrat = EmbedStrat.none # * we're not planning on going multitask in near future, so please hold.
    task_embed_size: int = 256
    task_embed_token_count: int = 1

    # This needs a separate API from the rest, likely, tied to readin.
    array_embed_strategy: EmbedStrat = EmbedStrat.none # ? maybe subsumed by subject
    array_embed_size: int = 256 # Bound in `propagate_config`

    active_assist_embed_strategy: EmbedStrat = EmbedStrat.none
    active_assist_embed_size: int = 256 # Bound in `propagate_config``

    passive_assist_embed_strategy: EmbedStrat = EmbedStrat.none
    passive_assist_embed_size: int = 256 # Bound in `propagate_config``

    # Closely related to, but not quite, array embed strategy.
    # Array embed strategy describes how we should provide information about array
    # Readin strategy describes IO.
    # Only when readin strategy is `token` does array embed get used.
    readin_strategy: EmbedStrat = EmbedStrat.token
    readin_dim: int = 32 # a multipurpose readin hidden size. Used differently in readin matrix and readin attention
    readin_compress: bool = True # factorize according to above dim
    readout_strategy: EmbedStrat = EmbedStrat.none
    readout_dim: int = 0 # use original space

    # Timestep level
    # "concat" becomes a valid strategy at this point
    stim_embed_strategy: EmbedStrat = EmbedStrat.token
    heldout_neuron_embed_strategy: EmbedStrat = EmbedStrat.token # Not even sure if there's a different way here.
    # There should maybe be a section for augmentation/ablation, but that is low pri.

    layer_norm_input: bool = False # layer norm on population input

    # Config for space-time. Control flows are not explicitly separated from base temporal transformer.
    transform_space: bool = False # master flag for space-time
    spike_embed_style: EmbedStrat = EmbedStrat.none # else - token (small), project (linear)
    spike_embed_dim: int = 0 # embedding dimension for spike counts (0 == infer as hidden size / neurons_per_token)
    neurons_per_token: int = 1 # how many neurons to embed per token (only makes sense for token/project)
    # This needs to match neurons_per_token in data config if data is in serve_tokenized mode
    max_neuron_count: int = 21 # pretty safe upper bound on number of neurons that can be embedded. Must be > data.pad_value
    max_return: int = 50 # max reward expected to embed or decode
    # We observe max is 13 in 15s trials (`proc_data_sampler`). Even if we rebin to 60ms bins and go to 45s, I doubt we'll go over 50; overhead of having a high max is low.

    causal: bool = True
    # autoregressive: bool = False # Stronger flag - does transformer only allow attending to literal previous tokens (For decoder only operations); not just in `time`

    log_backbone_norm: int = 0 # 1 for basic, 2 or higher not implemented
    log_token_seen_throughput: bool = False # for flat models - log post-crop non-padding tokens
    log_token_proc_throughput: bool = False # for flat models - log tokens
    # * ^ the above logs are actually going to be cumulative tokens processed, not throughput
    # realized that true wall clock fair tests are likely inconsistent for our tiny heterogeneous cluster

    debug_project_space: bool = False # project spikes for spacetime models to hidden size (only for very special cases, used in NLB parity)
    force_zero_mask: bool = False # for shuffle infill
    val_iters: int = 1 # how many iters to run validation for, since it's quite noisy for Pitt decode

    closed_loop_crop_bins: int = 0 # take last N bins for closed loop. For stability
    extra_task_embed_ckpt: str = "" # for loading task embeddings from a different ckpt. Only implemented via `model_decode`.
    extra_subject_embed_ckpt: str = "" # for loading subject embeddings from a different ckpt. Only implemented via `model_decode`.

    eval: EvalConfig = field(default_factory=EvalConfig)

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
    firing_hz_floor: float = 0.5
    minmax: bool = True # rescale kinematics to -1, 1
    chop_size_ms: int = 15000 # Not universally used but enough that I'm putting it for NDT3

    def reproc_dict(self) -> Dict[str, List[str]]:
        r"""
            Dictionary of attrs that should trigger a reprocessing events
        """
        return {}

    @classmethod
    def create_with_arrays(cls, arrays: List[str], **kwargs):
        return cls(arrays=arrays, **kwargs)

@dataclass
class RTTConfig(ExperimentalConfig):
    load_covariates: bool = True
    include_sorted: bool = False

    sampling_rate: int = 1000 # static
    covariate_sampling_rate: int = 250

    def reproc_dict(self):
        return {'chop_size_ms': self.chop_size_ms, 'include_sorted': self.include_sorted}

@dataclass
class MazeConfig(ExperimentalConfig):
    chop_size_ms: int = 15000 # no chop
    load_covariates: bool = False
    pretrial_time_s: float = 0.25
    posttrial_time_s: float = 0.1

    def reproc_dict(self):
        return {
            'chop_size_ms': self.chop_size_ms,
            'pretrial_time_s': self.pretrial_time_s,
            'posttrial_time_s': self.posttrial_time_s,
        }

@dataclass
class DyerCOConfig(ExperimentalConfig):
    load_covariates: bool = True
    velocity_threshold: float = 5. # Defunct

@dataclass
class NLBConfig(ExperimentalConfig):
    heldout_neurons: int = 32 # for RTT

@dataclass
class PittConfig(ExperimentalConfig):
    chop_size_ms: int = 2500
    respect_trial_boundaries: bool = False # keep this off for simplicity
    closed_loop_intention_estimation: str = ""
    limit_kin_dims: int = 8 # First 8 dims are taken (historically idx 6 is grasp velocity, 7 is grasp force)
    # clip_kinematics: float = 10.0 # we don't expect values outside this range. Something abberant is happening if we do, clip these.


@dataclass
class DatasetConfig:
    root_dir: Path = Path("./data")
    preprocess_suffix: str = 'preprocessed'

    # if number of trials below this, try loading into memory to accelerate tuning
    # if 0, ignores.
    auto_in_memory_thresh: int = 1000

    dataset_seed: int = 0 # for shuffling/splitting etc
    r"""
        Specifies the source datasets.
        - datasets accepts lists of strings that point to registered data files; this pointer can be one of:
            - paths to data files themselves
            - aliases (from registration)
            - lightweight regex for _aliases_ (not paths). Note this is regex, not glob.
    """
    datasets: List[str] = field(default_factory=lambda: [])
    exclude_datasets: List[str] = field(default_factory=lambda: []) # more specific aliases to exclude, processed after above, and no-ops for anything in `eval_datasets`
    data_blacklist: str = '' # path to text file with one dataset alias per line to exclude (for a first pass, above is more specific)

    scale_ratio: float = 1. # ratio of dataset to use for training (For scaling experiments)
    scale_limit: int = 0 # >0, limit number of trials (For scaling experiments). Mutually exclusive and override `scale_ratio`
    scale_limit_per_session: int = 0 # >0, limit number of trials per session (For scaling experiments)
    scale_limit_per_eval_session: int = 0 # >0, separately limit number of eval sessions (For scaling experiments)

    # Datasets to hold a _subset_ of from training. (some exposure still required)
    # These datasets are used for evaluation (in analysis, and possibly during training), separate from validation step.
    eval_datasets: List[str] = field(default_factory=lambda: [])
    eval_ratio: float = 1.0 # ratio of eval dataset to reserve for eval
    # eval_ratio: float = 0.5 # ratio of eval dataset to reserve for eval
    eval_force_limit: bool = False # if true, ignore eval ratio, and simply reserve reserve the above `scale_limit_per_session``.
    eval_seed: int = 0 # for shuffling/splitting etc

    # TODO what we really need are a suite of eval tasks/callbacks...

    eval_split_continuous: bool = False # For comparison with rEFH - make eval a continuous block that comes later in training.

    r"""
        `data_keys` and `meta_keys` specify the attributes of the dataset are served.
    """
    data_keys: List[DataKey] = field(
        default_factory=lambda: [DataKey.spikes]
    )
    meta_keys: List[MetaKey] = field(
        default_factory=lambda: [MetaKey.unique, MetaKey.session, MetaKey.array]
    ) # JY recommends providing array meta info, but thinks the system should be designed to not error without.

    heldout_key_spoof_shape: List[int] = field(default_factory=lambda: []) # spoof shape for heldout key if not available

    split_key: MetaKey = MetaKey.unique
    # ==== Data parsing/processing ====
    bin_size_ms: int = 2
    pad_batches: bool = True # else, trim batches to the shortest trial
    max_trial_length: int = 1500 # in bins. for preproc
    max_length_ms: int = 0 # in ms, in dataloader

    z_score: str = "" # path to dict with <session/alias> - decode normalizing things. Also generated by `data_kin_global_stat`. For behavior
    # each data stream should provide zscore values, if not, register will apply global defaults in base naive format. Just a working flag for NDT2-ish experiments.
    z_score_default_mean: float = 0.
    z_score_default_std: float = 1. # removed after minmax normalization became the norm

    augmentations: List[str] = field(default_factory=lambda: [])
    randaug_num: int = 1
    # list of augmentations during dataloading.

    # options: "" no z-scoring, session, global. See also model layer norm on input

    # Pad to this number of channels per array group
    # If set to 0, will skip padding checks.
    max_channels: int = 0 # ! TODO add smart inference (take max over array reports)

    # Pad to this number of arrays (for meta and data alike). Must be >= 1
    max_arrays: int = 1
    behavior_dim: int = 2

    tokenize_covariates: bool = False # Global preproc req. Should significantly change proc in CovariateReadout
    # Experimental config to test for signs of life in multimodal case.
    semantic_positions: bool = False # If covariates are tokenize, reserve specific dims for specific semantics (makes most sense in ctx of Pitt only exps)
    pad_positions: bool = False # Pad to global number of positions; for debugging and only pads to explicit DEFAULT_KIN_LABELS

    sparse_constraints: bool = False
    sparse_rewards: bool = False
    return_horizon_s: float = 10. # lookahead for return computation

    serve_tokenized: bool = False # master flag for space time operator (in anticipation that space time will move to tokenized)
    # Tokenized == serve B T S H instead of B T A C H
    serve_tokenized_flat: bool = False # flatten space (serve spikes as B Token H instead of B T S H)
    neurons_per_token: int = 8 # for tokenized
    max_tokens: int = 1024 # for tokenized - note we will still respect max_length_ms (limit fills in space and then either this inferred time limit or the explicit one)
    # This will be the # of tokens served; be generous because we will crop in any flat task.
    # ! note that the above is going to be strictly more than amount proc-ed in encoder-decoder encoder -- since things are cropped.
    pad_value: int = 0
    # pad_time_value defaults to max trial length (in bins)
    # pad_time_value: int = 400 # some reasonably high number to ensure we don't accidentally get padding tokens with padded time that can't attend to anything, but not so high that we're out of time range
    pad_spike_value: int = 0 # extra thing just for spikes, which we can typically afford to keep low w/o consequence. Sometimes above pad value (which applies for time/space values) needs to be set higher than 0 to avoid nan attn, typically for co-bps
    # pad_value: int = 20
    pad_position_value: int = 32 # Should match maximal space tokens per timestep (across all modalities) for next step prediction specifically

    # Experimental Task configuration - matching registered names
    # Note - we choose to put task specific things here rather than ModelConfig as model will read the relevant variables
    # from `data_attrs`. Tasks may be specified to e.g. load specific subsets of targets rather than full data
    # and so Dataset must know about this; and probably better to propagate this to ModelConfig than to have
    # to track it in two places.
    nlb_maze: NLBConfig = field(default_factory=NLBConfig)
    nlb_rtt: NLBConfig = field(default_factory=NLBConfig)
    churchland_maze: MazeConfig = field(default_factory=MazeConfig)
    odoherty_rtt: RTTConfig = field(default_factory=RTTConfig)
    dyer_co: DyerCOConfig = field(default_factory=DyerCOConfig)
    gallego_co: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    churchland_misc: ExperimentalConfig = field(default_factory=ExperimentalConfig)

    marino_batista_mp_bci: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    marino_batista_mp_iso_force: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    marino_batista_mp_reaching: ExperimentalConfig = field(default_factory=ExperimentalConfig)

    miller: ExperimentalConfig = field(default_factory=ExperimentalConfig)

    pitt_co: PittConfig = field(default_factory=lambda: PittConfig.create_with_arrays([ # This is actually the catch all for Pitt, and doesn't have any particular structure. No guarantees, might not even be CO.
        'CRS02b-lateral_m1', 'CRS02b-medial_m1',
        'CRS07-lateral_m1', 'CRS07-medial_m1',
        'CRS08-lateral_m1', 'CRS08-medial_m1',
    ]))

    observation: PittConfig = field(default_factory=lambda: PittConfig.create_with_arrays([
        'CRS02b-lateral_m1', 'CRS02b-medial_m1',
        'CRS07-lateral_m1', 'CRS07-medial_m1',
        'CRS08-lateral_m1', 'CRS08-medial_m1',
    ]))

    ortho: PittConfig = field(default_factory=lambda: PittConfig.create_with_arrays([
        'CRS02b-lateral_m1', 'CRS02b-medial_m1',
        'CRS07-lateral_m1', 'CRS07-medial_m1',
        'CRS08-lateral_m1', 'CRS08-medial_m1',
    ]))

    fbc: PittConfig = field(default_factory=lambda: PittConfig.create_with_arrays([
        'CRS02b-lateral_m1', 'CRS02b-medial_m1',
        'CRS07-lateral_m1', 'CRS07-medial_m1',
        'CRS08-lateral_m1', 'CRS08-medial_m1',
    ]))

    unstructured: PittConfig = field(default_factory=lambda: PittConfig.create_with_arrays([
        'CRS02b-lateral_m1', 'CRS02b-medial_m1',
        'CRS07-lateral_m1', 'CRS07-medial_m1',
        'CRS08-lateral_m1', 'CRS08-medial_m1',
    ]))
    delay_reach: ExperimentalConfig = field(default_factory=ExperimentalConfig)

    permute_channels: bool = False # test flag, permute channels randomly per session

@dataclass
class TrainConfig:
    epochs: int = 10000
    steps: int = 0 # Prefer to specify steps over epochs for FLOP consistency (pretty loose), but most other training settings are on epochs
    log_every_n_steps: int = 10
    batch_size: int = 0
    effective_batch_size: int = 512
    patience: int = 50 # these are in units of val checks (epochs)
    log_grad: bool = False
    gradient_clip_val: float = 1.0
    accumulate_batches: int = 1
    autoscale_batch_size: bool = True
    max_batch_size: int = 4096 # if autoscale, this is the max batch size
    overfit_batches: bool = False
    profiler: str = ""
    val_check_epochs: int = 1
    val_check_interval: int = 0 # these are in steps, but mostly isn't used
    strategy: str = "" # uses DDP or auto by default, can specify deepspeed

@dataclass
class RootConfig:
    seed: int = 0
    tag: str = "" # i.e. experiment variant, now an optional tag (since hydra consumes file, we can't use the filename for experiment name. Specify if you want.)
    experiment_set: str = ""
    # for wandb
    notes: str = ""
    trainable_parameters: int = 0
    total_parameters: int = 0

    # Meta config - will initiate multiple derivative runs, all handled in `run.py`
    sweep_cfg: str = "" # See `hp_sweep_space.py`
    sweep_trials: int = 8
    sweep_mode: str = 'random' # or grid, which is implicitly exhaustive
    sweep_tag: str = "" # * don't specify this, we use this to track in wandb

    fragment_datasets: bool = False # splits run into multiple runs, one per dataset. For single-session baselines

    default_root_dir: Path = Path("./data/runs").resolve()
    wandb_user: str = "joelye9"
    wandb_project: str = "ndt3"
    wandb_api_key_path: Path = Path("/home/joelye/.wandb_api").resolve()
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Initialization

    # wandb ids
    init_from_id: str = "" # for initializing weights
    init_ckpt: str = "" # fallback for above, for portable runs
    load_from_id: str = "" # for resuming training. takes precedent over init_from_id
    init_tag: str = "val_loss"

    # orchestration
    inherit_exp: str = "" # hunt wandb for the relevant experiment, presumed same tag name.
    inherit_tag: str = "" # override same tag inheritance
    serial_run: bool = False # for launchers..

    cancel_if_run_exists: bool = True # since codebase is fairly stable now - if same config/tag exists on wandb, do not run.
    # Only checked if `inherit_exp` is set i.e. part of chain of runs. See `ckpts_and_wandb_helpers/wandb_run_exists`

    successor_exp: List[str] = field(
        default_factory=lambda: []
    ) # if set, will run this experiment after this one finishes. See `ckpts_and_wandb_helpers/wandb_run_exists

    # use_ckpt_model_cfg: bool = False


    probe_finetune: bool = False # If true, fit probe (novel params unlocked and trained), and then unfreeze, reset to best val, and train the rest of the model. Same training params are used in both instanced.
    # See https://arxiv.org/pdf/2202.10054.pdf (In pilots, not useful, deprecated)

    exp: Any = MISSING # delta config, provide via yaml and on CLI as `+exp=<test>.yaml`
    slurm_id: int = 0 # for experiment tracking...
    nodes: int = 1
    debug: bool = False # for debugging, don't log to wandb, don't save ckpts, etc

BatchKey = str | DataKey | MetaKey

def propagate_config(config: RootConfig):
    r"""
        There wasn't an obvious way to bind configuration across sub-nodes (even if that has bad code-smell, we often use it).
        We patch that here.
        This step only needs to happen when we read from a YAML, i.e. wandb should only store propagated versions.
    """
    config.dataset.neurons_per_token = config.model.neurons_per_token
    assert config.model.transformer.max_trial_length >= config.dataset.max_trial_length, \
        f"max_trial_length {config.model.transformer.max_trial_length} in model must exceed that served by dataset {config.dataset.max_trial_length}"
    # config.model.transformer.max_trial_length = config.dataset.max_trial_length

    config.model.transformer.n_state = config.model.hidden_size
    config.model.transformer.dropout = config.model.dropout
    config.model.transformer.transform_space = config.model.transform_space
    config.model.session_embed_size = config.model.hidden_size
    config.model.subject_embed_size = config.model.hidden_size
    config.model.array_embed_size = config.model.hidden_size
    config.model.task_embed_size = config.model.hidden_size
    config.model.active_assist_embed_size = config.model.hidden_size
    config.model.passive_assist_embed_size = config.model.hidden_size

    config.model.readin_dim = config.model.hidden_size
    config.model.readout_dim = config.model.hidden_size
    config.model.task.decode_tokenize_dims = config.dataset.tokenize_covariates