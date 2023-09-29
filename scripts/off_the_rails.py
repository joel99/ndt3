#%%
# Show individual predictions with respect to time/cue time.
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
import seaborn as sns
import torch
torch.set_warn_always(False) # Turn off warnings, we get cast spam otherwise
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from context_general_bci.model import transfer_model
from context_general_bci.dataset import SpikingDataset
from context_general_bci.config import RootConfig, ModelConfig, ModelTask, Metric, Output, DataKey, MetaKey
from context_general_bci.contexts import context_registry
from context_general_bci.config import REACH_DEFAULT_KIN_LABELS, REACH_DEFAULT_3D_KIN_LABELS
from context_general_bci.tasks.myow_co import DYER_DEFAULT_KIN_LABELS
from context_general_bci.tasks.miller import MILLER_LABELS

from context_general_bci.analyze_utils import stack_batch, load_wandb_run, prep_plt
from context_general_bci.utils import get_wandb_run, wandb_query_latest

query = 'rtt-gvgaiv76'
# query = 'bhvr_12l_512-ijdvhprq'
# query = 'base-2dvz5mgm'
# query = 'rtt_c512-fr4nb2hw'
# query = 'rtt_c512_m5-83uzsg6w'
# query = 'base-1hw7fz9e' # Miller
query = 'rtt_c512_bsz_256-ji1wdvmg'
# query = 'rtt_c512_m8_bsz_256-c7ldp7xm'
# query = 'rtt_c512_km8_bsz_256-2vx7pj8e'
query = 'monkey_c512_km8_bsz_256-x5y1sfpa'
query = 'bhvr_12l_512_km8_c512-abij2xtx' # currently -0.36, lol.
query = 'monkey_trialized-5qp70fgs'
# query = 'bhvr_12l_1024_km8_c512-6p6h9m7l'
# query = 'monkey_trialized-peu3ln1l'
# query = 'monkey_trialized_6l_1024-22lwlmk7'
query = 'monkey_trialized_6l_1024-zgsjsog0'

# query = 'monkey_trialized_6l_1024_broad-3x3mrjdh'
# query = 'monkey_trialized_6l_1024_broad-yy3ve3gf'
# query = 'monkey_trialized_6l_1024_all-ufyxs032'

# query = 'monkey_schedule_6l_1024-zfwshzmr'
# query = 'monkey_schedule_6l_1024-0swiit7z'
query = 'monkey_kin_6l_1024-vgdhzzxm'
query = 'monkey_random_6l_1024-n3f68hj2'

query = 'monkey_random_6l_1024_d2-s1wrxq2e'

wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_loss')

# cfg.model.task.metrics = [Metric.kinematic_r2]
cfg.model.task.outputs = [Output.behavior, Output.behavior_pred]

target = [
    # 'miller_Jango-Jango_20150730_001',

    # 'dyer_co_.*',
    # 'dyer_co_mihi_1',
    # 'gallego_co_Chewie_CO_20160510',
    # 'churchland_misc_jenkins-10cXhCDnfDlcwVJc_elZwjQLLsb_d7xYI',
    # 'churchland_maze_nitschke-sub-Nitschke_ses-20090812',
    # 'churchland_maze_jenkins.*'
    # 'odoherty_rtt-Indy-20160407_02',
    # 'odoherty_rtt-Indy-20160627_01',
    # 'odoherty_rtt-Indy-20161005_06',
    # 'odoherty_rtt-Indy-20161026_03',
    # 'odoherty_rtt-Indy-20170131_02',
    # 'odoherty_rtt-Indy-20160627_01',
    # 'odoherty_rtt-Loco-20170210_03',
    # 'odoherty_rtt-Loco-20170213_02',
    # 'odoherty_rtt-Loco-20170214_02',
    'odoherty_rtt-Loco-20170215_02',
    'odoherty_rtt-Loco-20170216_02',
    'odoherty_rtt-Loco-20170217_02'
]

cfg.dataset.datasets = target
dataset = SpikingDataset(cfg.dataset)
pl.seed_everything(0)
dataset.subset_scale(limit_per_session=48)
print("Eval length: ", len(dataset))
data_attrs = dataset.get_data_attrs()
print(data_attrs)

model = transfer_model(src_model, cfg.model, data_attrs)

# model.cfg.eval.teacher_timesteps = int(50 * 0.) # 0.5s
# model.cfg.eval.teacher_timesteps = int(50 * 0.5) # 0.5s
# model.cfg.eval.teacher_timesteps = int(50 * 0.5) # 0.5s
model.cfg.eval.teacher_timesteps = int(50 * 2) # 2s
# model.cfg.eval.teacher_timesteps = int(50 * 0.1) # 0.5s
model.cfg.eval.limit_timesteps = 50 * 4 # up to 4s
model.cfg.eval.temperature = 0.
# model.cfg.eval.temperature = 0.01

def get_eval(model, dataset, batch_size=48):
    trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
    def get_dataloader(dataset: SpikingDataset, batch_size=batch_size, num_workers=1, **kwargs) -> DataLoader:
        return DataLoader(dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            collate_fn=dataset.tokenized_collater,
        )

    dataloader = get_dataloader(dataset)
    return stack_batch(trainer.predict(model, dataloader))

model.cfg.eval.use_student = False
no_student = get_eval(model, dataset)
model.cfg.eval.use_student = True
with_student = get_eval(model, dataset)

#%%
DIMS = {
    'gallego': REACH_DEFAULT_KIN_LABELS,
    'dyer': DYER_DEFAULT_KIN_LABELS,
    'miller': MILLER_LABELS,
    'churchland_misc': REACH_DEFAULT_3D_KIN_LABELS,
    'churchland_maze': REACH_DEFAULT_KIN_LABELS,
    'delay': REACH_DEFAULT_3D_KIN_LABELS,
    'odoherty': REACH_DEFAULT_KIN_LABELS,
}
data_label = [i for i in DIMS.keys() if dataset.cfg.datasets[0].startswith(i)][0]
print(f'Assuming: {data_label}')
from sklearn.metrics import r2_score
print(no_student[Output.behavior].shape)


def rolling_time_since_student(bool_tensor):
    # bool_tensor: 1D, false is teacher, true if student. Used to identify "going off the rails"
    # Identify change points
    change_points = (bool_tensor[:-1] & ~bool_tensor[1:]).nonzero(as_tuple=True)[0] + 1

    # Compute cumsum
    result = torch.cumsum(bool_tensor.int(), dim=0)

    # Adjust tail values based on change points
    for idx in change_points:
        result[idx:] -= result[idx-1]

    return result, change_points
import torch.nn.functional as F

def get_mse(outputs):
    prediction = outputs[Output.behavior_pred]
    target = outputs[Output.behavior]
    is_student = outputs[Output.behavior_query_mask]

    is_student_rolling, trial_change_points = rolling_time_since_student(is_student)
    mse = F.mse_loss(target, prediction, reduction='none')
    return is_student_rolling, mse

no_student_rolling, no_student_mse = get_mse(no_student)
with_student_rolling, with_student_mse = get_mse(with_student)
import pandas as pd
df = pd.DataFrame({
    'mse': torch.cat([no_student_mse, with_student_mse]),
    'cue_rolling': torch.cat([no_student_rolling, with_student_rolling]),
    'student_forced': torch.cat([torch.zeros_like(no_student_rolling), torch.ones_like(with_student_rolling)]),
})

df_wide = pd.DataFrame({
    'with_student': with_student_mse,
    'no_student': no_student_mse,
    'Tokens since teacher': with_student_rolling,
})

#%%
x_mode = 'relative'
x_mode = 'abs'
df['cue_abs'] = model.cfg.eval.teacher_timesteps * len(DIMS[data_label]) + df['cue_rolling']
f = plt.figure(figsize=(10, 10))
ax = prep_plt(f.gca(), big=True)

# sns.scatterplot(data=df_wide, x='with_student', y='no_student', hue='Tokens since teacher', ax=ax, s=5, alpha=0.8)
# ax.set_xlabel('With student MSE')
# ax.set_ylabel('No student MSE')


sns.lineplot(data=df, x='cue_abs' if x_mode == 'abs' else 'cue_rolling', y='mse', hue='student_forced', ax=ax)
# sns.scatterplot(data=df, x='cue_rolling', y='mse', hue='student_forced', ax=ax, s=3, alpha=0.4)
ax.set_ylabel("MSE")
ax.set_xlabel('Token pos' if x_mode == 'abs' else 'Tokens since teacher')

ax.set_title(f'{query} (Data: {data_label}) Teacher: {model.cfg.eval.teacher_timesteps} steps')
print(no_student_mse.mean())
print(with_student_mse.mean())
ax.set_xlim(0, 400)
ax.set_ylim(0, 0.015)

#%%
differential = no_student_mse - with_student_mse
f = plt.figure(figsize=(10, 10))
ax = prep_plt(f.gca(), big=True)
ax.scatter(no_student_rolling, differential, s=3, alpha=0.4, label='No Student vs Student in Time')
ax.set_xlabel('Tokens since teacher')
ax.set_ylabel('No Student - Student MSE difference')
