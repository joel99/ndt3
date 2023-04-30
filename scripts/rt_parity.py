#%%
# Compression was rapid. Now make sure the outputs are the same...
from pathlib import Path
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
import numpy as np
import torch

import seaborn as sns
import pandas as pd
import pytorch_lightning as pl
from einops import rearrange

# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.dataset import SpikingDataset, DataAttrs
from context_general_bci.config import ModelTask, Output, DataKey

from context_general_bci.analyze_utils import stack_batch, load_wandb_run
from context_general_bci.analyze_utils import prep_plt, get_dataloader
from context_general_bci.utils import wandb_query_experiment, get_wandb_run, wandb_query_latest


parity_mode = 'old'
parity_mode = 'new'
if parity_mode == 'old':
    from context_general_bci.model import transfer_model
else:
    from context_general_bci.model_decode import transfer_model
pl.seed_everything(0)

run_id = 'human-sweep-simpler_lr_sweep-89111ysu'
run_id = 'human_m5-s3n89xxv'
run_id = 'human_fbc-0epmqhls'
# run_id = 'human_rtt_pitt_init-idcfk3rr'
# run_id = 'human_aug-nxy3te61'
# run_id = 'human_aug-xi7wzqoo'

dataset_name = 'observation_CRS02b_19.*'
# dataset_name = 'odoherty_rtt-Indy-20160627_01'

run = get_wandb_run(run_id)
src_model, cfg, data_attrs = load_wandb_run(run, tag='val_loss')
cfg.dataset.datasets = [dataset_name]
cfg.dataset.eval_datasets = [dataset_name]
cfg.dataset.exclude_datasets = []
dataset = SpikingDataset(cfg.dataset)
dataset.subset_split(splits=['eval'])
dataset.build_context_index()
data_attrs = dataset.get_data_attrs()
cfg.model.task.tasks = [ModelTask.kinematic_decoding]
cfg.model.task.outputs = [Output.behavior_pred]
model = transfer_model(src_model, cfg.model, data_attrs)
model.eval()
dataloader = get_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0)

# Setup timing harness
import time

loop_times = []
# mode = 'cpu'
mode = 'gpu'
compile_flag = ''
# compile_flag = 'torchscript'
# compile_flag = 'onnx'
# onnx_file = 'model.onnx'

if mode == 'gpu':
    model = model.to('cuda:0')

if compile_flag == 'torchscript':
    model = model.to_torchscript()

if compile_flag == 'onnx' and Path(onnx_file).exists():
    import onnxruntime
    ort_session = onnxruntime.InferenceSession(onnx_file)
    input_name = ort_session.get_inputs()[0].name
    input_sample = torch.load('samples.pt')
    ort_inputs = {input_name: input_sample}
    do_onnx = True
else:
    do_onnx = False

loops = 50
pl.seed_everything(0)

# trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
# trainer_out = trainer.predict(model, dataloader)
# trainer_out = stack_batch(trainer_out)
# # import pdb;pdb.set_trace()
# # Recast for trainer...
# if mode == 'gpu':
#     model = model.to('cuda:0')

test_ins = []
test_outs = []
backbone_payloads = []

# ? Stereotypy may be built into the task -- if we tell the model we're using RTT, maybe it won't be stereotyped, but it is during Pitt?
with torch.no_grad():
    # for i in range(50):
    for batch in dataloader:
    # for trial in dataset:
        # import pdb;pdb.set_trace()
        if parity_mode == 'new':
            spikes = rearrange(batch[DataKey.spikes], 'b (time space) chunk 1 -> b time (space chunk) 1', space=6)
            # equivalent to loading a single trial for Pitt data.
        # spikes = trial[DataKey.spikes].flatten(1,2).unsqueeze(0) # simulate normal trial
        spikes = torch.randint(0, 2, (1, 100, 192, 1), dtype=torch.uint8)
        test_ins.append(spikes)
        start = time.time()
        if do_onnx:
            out = ort_session.run(None, ort_inputs)
        else:
            if mode == 'gpu':
                if parity_mode == 'new':
                    spikes = spikes.to('cuda:0')
                else:
                    for k in batch:
                        batch[k] = batch[k].to('cuda:0')

            if parity_mode == 'new':
                # print('testing rand in')
                out = model(spikes)

            if parity_mode == 'old':
                out = model(batch)
                out = model.task_pipelines[ModelTask.kinematic_decoding.value](
                    batch,
                    out,
                    compute_metrics=False,
                    eval_mode=True
                )[Output.behavior_pred]

            if mode == 'gpu':
                out = out.to('cpu')
        end = time.time()
        test_outs.append(out)
        # print(out)
        if compile_flag == 'onnx' and not Path(onnx_file).exists():
            spikes = torch.randint(0, 4, (1, 100, 192, 1), dtype=torch.uint8)
            model = model.to_onnx("model.onnx", spikes, export_params=True)
            torch.save(spikes, 'samples.pt')
            exit(0)
        loop_times.append(end - start)
        # print(f'Loop: {end - start:.4f}')
        # print(f'Loop {spikes.size()}: {end - start:.4f}')
# drop first ten
loop_times = loop_times[10:]

# print(f'Benchmark: {run_id}. Data: {dataset_name}')
print(f'Benchmark: {mode}')
print(f"Avg: {np.mean(loop_times)*1000:.4f}ms, Std: {np.std(loop_times) * 1000:.4f}ms")

# Key check
# Trial context, time, position is matched
# State in? Backbone?

#%%
stat_test = torch.stack(test_outs)
print(stat_test.shape)
sns.histplot(stat_test.flatten())
print(model.decoder.bhvr_std)
print(stat_test.flatten().std())

#%%
# plot outputs
# trial = 0
# trial = 1
trial = 2
# trial = 20
# trial = 10
# trial = 1
ax = prep_plt()
if parity_mode == 'new':
    trial_vel = test_outs[trial].numpy()
if parity_mode == 'old':
    trial_vel = test_outs[trial][0].numpy()
print(trial_vel.shape)
# trial_vel = trial_vel[:47]
# trial_vel = trainer_out[Output.behavior_pred][0].numpy()
for i in range(trial_vel.shape[1]):
    ax.plot(trial_vel[:,i][2:])
    # ax.plot(trial_vel[:,i].cumsum())
ax.set_ylim(-1, 1)
ax.set_title(f'Velocity random inject {parity_mode}') #  {trial_vel.shape}')

#%%
# Make a grid that plots 9 trials
import matplotlib.pyplot as plt
import numpy as np

def plot_trial_velocities(trial_vel, parity_mode):
    fig, ax = plt.subplots()

    for i in range(trial_vel.shape[1]):
        ax.plot(trial_vel[:, i][2:])
        # ax.plot(trial_vel[:, i][2:])

    ax.set_title(f'Velocity random inject {parity_mode}')
    plt.show()

def create_grid_plot(test_outs, trials):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.tight_layout(pad=4)

    for i, ax in enumerate(axes.flat):
        trial = trials[i]

        if parity_mode == 'new':
            trial_vel = test_outs[trial].numpy()
        if parity_mode == 'old':
            trial_vel = test_outs[trial][0].numpy()

        for j in range(trial_vel.shape[1]):
            ax.plot(trial_vel[:, j][2:])

        ax.set_title(f'Velocity random inject {parity_mode} (Trial {trial})')
        ax.set_ylim(-3, 3)

    plt.show()

# Replace this part with your actual data
trials = [0, 1, 2, 3, 4, 5, 6, 7, 8]
trials = np.linspace(0, 49, 9, dtype=int)

create_grid_plot(test_outs, trials)


#%%
torch.save(trial_vel, f'trial_vel_{parity_mode}.pt')
torch.save({
    'in': test_ins,
    'out': test_outs
}, 'onnx_parity.pt')
#%%

print(backbone_payloads[0][0].shape)
if parity_mode == 'new':
    trial_backbone = backbone_payloads[0].cpu().numpy()
else:
    trial_backbone = backbone_payloads[0][0].cpu().numpy()
for i in range(10):
# for i in range(trial_backbone.shape[1]):
    plt.plot(trial_backbone[:10,i])