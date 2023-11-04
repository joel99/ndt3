#%%
# It's hard to visualize the effect of parameters on full training curves in wandb.
# Import necessary libraries
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize a wandb run
# wandb.init(project="ndt3", entity="joel99")
from context_general_bci.utils import wandb_query_latest
from context_general_bci.analyze_utils import prep_plt

filter_kwargs = {
    "config.experiment_set": "sixty/tune2"
}
exp = "rouse-sweep"
exp = "indy_miller-sweep"
exp = "crs08-sweep"
runs = wandb_query_latest(exp, **filter_kwargs)


# Function to plot validation curves for a set of runs
def plot_validation_curves(runs):
    # Create empty dataframe to hold all run histories
    all_run_histories = []

    metric = 'infill_loss'
    metric = 'r2'

    # Extract run histories
    for run in runs:
        # Pull down the validation metrics for the run
        history = run.scan_history(keys=[f'val_kinematic_{metric}', f'eval_kinematic_{metric}'])
        history_df = pd.DataFrame(history)
        history_df['run_name'] = run.name
        history_df['lr_init'] = run.config['model']['lr_init']
        history_df['lr_schedule'] = run.config['model']['lr_schedule']

        # Append to the all_run_histories dataframe
        all_run_histories.append(history_df)
    all_run_histories = pd.concat(all_run_histories)
    # Plot using seaborn
    plt.figure(figsize=(18, 6))

    # Validation Loss Plot
    ax = plt.subplot(1, 3, 1)
    prep_plt(ax)
    sns.lineplot(data=all_run_histories, x=all_run_histories.index, y=f'val_kinematic_{metric}',
                 hue='lr_init', style='lr_schedule', markers=True, dashes=False, alpha=0.8)
    # plt.title('Val loss')
    plt.xlabel('Epochs')
    plt.ylabel(f'Val {metric}')

    # Validation Accuracy Plot
    ax = plt.subplot(1, 3, 2)
    prep_plt(ax)
    sns.lineplot(data=all_run_histories, x=all_run_histories.index, y=f'eval_kinematic_{metric}',
                 hue='lr_init', style='lr_schedule', markers=True, dashes=False,
                 alpha=0.8)
    # plt.title('Eval loss')
    plt.xlabel('Epochs')
    plt.ylabel(f'Eval {metric}')

    # Scatter
    ax = plt.subplot(1, 3, 3)
    prep_plt(ax)
    sns.scatterplot(data=all_run_histories, x=f'val_kinematic_{metric}', y=f'eval_kinematic_{metric}',
                    hue='lr_init', style='lr_schedule',
                    alpha=1.0)
    # Plot y=x in the coordinate system
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()
    min_xy = min(min_x, min_y)
    max_xy = max(max_x, max_y)
    ax.plot([min_xy, max_xy], [min_xy, max_xy], ls="--", c=".3")
    ax.set_xlim(min_xy, max_xy)
    ax.set_ylim(min_xy, max_xy)

    plt.suptitle(exp)
    plt.tight_layout()
    plt.show()
plot_validation_curves(runs)
