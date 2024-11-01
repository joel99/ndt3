#%%
# Main plot on RTT occlusion is generated by manual stitching of `plot_subject_occlusion.py` and `plot_occlusion.py`
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import seaborn as sns
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import os
import subprocess
import time

# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.plotting import prep_plt, colormap, MARKER_SIZE

eval_set = 'rtt'
csvs = [
    'data/analysis_metrics/rtt_session_occ.csv',
    'data/analysis_metrics/rtt_shuffle_semitoken_session_occ.csv',
    'data/analysis_metrics/rtt_shuffle_token_session_occ.csv',
    'data/analysis_metrics/rtt_shuffle_channel_session_occ.csv',
]
df = pd.concat([pd.read_csv(csv) for csv in csvs])

print(df)
def stem_map(variant):
    if 'transfer' in variant:
        if 'scratch' in variant:
            stem = 'scratch_transfer'
        else:
            stem = '_'.join(variant.split('-')[0].split('_')[:-1]) + '_transfer'
    else:
        stem = '_'.join(variant.split('-')[0].split('_')[:-1])
    return stem

def day_of(variant):
    if '2s' in variant:
        return 2
    elif '5s' in variant:
        return 5
    elif '10s' in variant:
        return 10
    if 'transfer' in variant:
        if 'scratch' in variant:
            day = variant.split('-')[0].split('_')[-1][:-1] # trim "d" from "0d"
        else:
            day = variant.split('-')[0].split('_')[-1][len('transfer'):-1] # trim "d" from "0d"
    else:
        day = variant.split('-')[0].split('_')[-1][:-1] # trim "d" from "0d"
    return int(day)

df['variant_stem'] = df.apply(lambda row: stem_map(row.variant), axis=1)
df['day'] = df.apply(lambda row: day_of(row.variant), axis=1)
df['is_subject'] = df.variant.apply(lambda x: '2s' in x or '5s' in x or '10s' in x)
day_to_time_map = {
    'rtt': {
        0: 0,
        4: 793 - 60,
        8: 1992 - 60, # Numbers are pulled from # of trials
        60: 4310 - 60,
        120: 9932 - 60,
        2: 4245 - 60,
        5: 11248 - 60,
        10: 20325 - 60,
    },
}
df['daytime_hr'] = df.apply(lambda row: day_to_time_map[eval_set][row.day] / 60 / 60, axis=1)
df['daytime_min'] = df.apply(lambda row: day_to_time_map[eval_set][row.day] / 60, axis=1)


target_df = df
#%%
# f = plt.figure(figsize=(3.2 if eval_set == 'rtt' else 2.75, 3.5), layout='constrained')
f = plt.figure(figsize=(2.75, 3.5), layout='constrained')
ax = prep_plt(f.gca(), big=True)

ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.yaxis.grid(True, which='minor', linestyle='-', linewidth=0.5, alpha=0.5)

subset_variant = [
    'scratch',
    'base_45m_200h',
    'big_350m_2kh',
    'scratch_transfer',
    'base_45m_200h_transfer',
    'big_350m_2kh_transfer',
]
y = 'eval_r2'

# exp_subset = 'rtt'
exp_subset = 'rtt_shuffle_token'
exp_subset = 'rtt_shuffle_semitoken'
# exp_subset = 'rtt_shuffle_channel'

SHOW_Y = exp_subset == 'rtt'

def marker_style_map(variant_stem):
    if '350m' in variant_stem:
        return 'P'
    if variant_stem in ['NDT2 Expert', 'NDT3 Expert', 'scratch', 'NDT3 mse', 'wf', 'ole', 'scratch_transfer']:
        return 'X'
    else:
        return 'o'
marker_dict = {
    k: marker_style_map(k) for k in target_df['variant_stem'].unique()
}
style_map = {
    'scratch': (5,0),
    'base_45m_200h': (5, 0),
    'base_45m_2kh': (5, 0),
    'big_350m_2kh': (5, 0),
    'scratch_transfer': (5, 2),
    'base_45m_200h_transfer': (5, 2),
    'big_350m_2kh_transfer': (5, 2),
}
x = 'daytime_hr' if eval_set == 'rtt' else 'daytime_min'
# x = 'daytime_min'
subset_df = target_df[target_df.variant_stem.isin(subset_variant)]

subset_df = subset_df[subset_df[x] > 0] # annotate as horizontal lines to connect panels with `plot_oclcusion`

def get_mean_perf(df):
    return df.groupby(['variant_stem', 'eval_set', 'day', 'daytime_hr', 'experiment_set', ]).agg({
        'eval_r2': 'mean',   # Take mean of eval_r2 across seeds
    }).reset_index()

subset_df = subset_df[subset_df['experiment_set'] == exp_subset]
session_df = subset_df[subset_df['is_subject'] == False]
subject_df = subset_df[subset_df['is_subject'] == True]
print(session_df.variant_stem.unique())
mean_df = get_mean_perf(session_df)

sns.lineplot(
        data=session_df,
        x=x,
        y=y,
        hue='variant_stem',
        palette=colormap,
        style='variant_stem',
        dashes=style_map,
        ax=ax,
        alpha=0.8, # Lighten it up
        err_kws={'alpha': 0.05},  # This makes the error band lighter
        # errorbar='sd',
    )

sns.scatterplot(
    data=mean_df,
    x=x,
    y=y,
    hue='variant_stem',
    palette=colormap,
    style='variant_stem',
    markers=marker_dict,
    ax=ax,
    s=MARKER_SIZE,
    legend=False,
    # legend=True,
    alpha=0.8,
)

# Scatter subject on top
mean_subject_df = get_mean_perf(subject_df)
sns.lineplot(
        data=subject_df,
        x=x,
        y=y,
        color='red',
        ax=ax,
        linestyle='--',
        alpha=0.8, # Lighten it up
        err_kws={'alpha': 0.05},  # This makes the error band lighter
        # errorbar='sd',
    )
sns.scatterplot(
    data=mean_subject_df,
    x=x,
    y=y,
    color='red',
    marker='X',
    s=MARKER_SIZE,
    # markers=marker_dict,
)

# ax.set_xscale('log')
# ax.set_xticks([])
# ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
# ax.xaxis.get_major_formatter().set_scientific(False)
# ax.ticklabel_format(style='plain', axis='x')
ax.legend().remove()
ax.set_ylabel("")
if not SHOW_Y:
    ax.set_yticklabels([])
# ax.annotate('$R^2$', xy=(0, 1), xytext=(-ax.yaxis.labelpad - 15, 1),
            # xycoords='axes fraction', textcoords='offset points',
            # ha='center', va='center', fontsize=24)
if eval_set == 'rtt':
    ax.set_xlabel('Cross-session data (hr)')
    ax.set_xlabel('') # Manual session join
elif eval_set == 'cursor':
    ax.set_xlabel('Cross-session data (min)')

ax.set_ylim(0.1, 0.8)
ax.set_yticks(np.arange(0.1, 0.8, 0.2))
# set minor ticks
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.set_xscale('log')
ax.set_xticks([0.5,1, 2])
ax.set_xticklabels([0.5, 1, 2])
ax.text(1.0, -0.03, 'hr', transform=ax.transAxes, ha='right', va='top', fontsize=22)

# hide every other tick

# ax.spines['left'].set_position(('axes', -0.05))  # Adjust as needed
from matplotlib.ticker import SymmetricalLogLocator

for variant in ['scratch', 'base_45m_200h', 'big_350m_2kh']:
    perf = target_df[(target_df[x] == 0) & (target_df['variant_stem'] == variant)][y].values[0]
    ax.axhline(perf, linestyle=':', color=colormap[variant], alpha=0.8)

# if eval_set == 'rtt':
#     ax.set_ylim(0.1, 0.8)
#     ax.set_yticks([0.3, 0.5, 0.7])
#     ax.set_xlim(-0.01, 10)
#     ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
#     ax.set_xscale('symlog', linthresh=0.1)
#     ax.xaxis.set_minor_locator(SymmetricalLogLocator(linthresh=0.1, base=10))

# Aesthetics todo
# Figure out overall layout / fontsize
# Insert variant_stem labels at start of plot

# %%

#%%
# Make an empty plot with annotation, $R^2$
f = plt.figure(figsize=(4, 3.5), layout='constrained')
ax = prep_plt(f.gca(), big=True)
# turn grid off
ax.grid(False)
ax.annotate('$R^2$', xy=(0.5, 0.5), xytext=(-ax.yaxis.labelpad - 15, 1),
            xycoords='axes fraction', textcoords='offset points',
            ha='center', va='center', fontsize=24)
plt.show()