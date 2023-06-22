import os
import shutil
import wandb
from pathlib import Path
# Initialize wandb
api = wandb.Api()

# Project name (change to your project name)
project_name = "context_general_bci"

# List of config.experiment_set values to filter
experiment_sets = [
    # "hetero", "nlb_v2", "pt_vs_ft", "pilot", "arch_rtt", "parity/bin20", "online_bci", "pilot/decode", "arch/tune_hp", "arch/tune_hp",
    "pitt_v2/decode", "odoherty/tune_mc_rtt", "", "odoherty/decode", "decode_sanity", "arch_rtt"
]

# Get all runs in the project
runs = api.runs(project_name)
# breakpoint()
# Filter runs
filtered_runs = [
    run for run in runs if 'test' in run.name or run.config.get('experiment_set') in experiment_sets \
]

# Base directory for checkpoints
base_dir = "./data/runs/"

# Iterate over filtered runs and delete checkpoints

for run in filtered_runs:
    run_dir = os.path.join(base_dir, 'context_general_bci', run.id)
    # Check if the directory exists
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
        print(f"Deleted checkpoints for run ID: {run.id}")
    else:
        print(f"No checkpoints found for run ID: {run.id}")
    # Check if logdir exists
    log_dir = (Path(base_dir)/ 'wandb').glob(f'*-{run.id}')
    try:
        log_dir = log_dir.__next__()
        shutil.rmtree(log_dir)
    except:
        print(f'skip logdir')
