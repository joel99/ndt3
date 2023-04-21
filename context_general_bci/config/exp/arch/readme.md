# Fig 1: Architecture superiority

Experiments for architecture comparisons. Derived from `saturation` line of exps.
Diffs:
- we use same test session as in data sourcing transfer experiments for consistency. (we picked `Indy-20160627_01` there for having a large number of trials and being relatively centered in time).
- control number of test session trials - keeping it relatively small (but still practically a functionally relevant "large calibration" regime) - 5 minutes; so as to amplify sample efficiency differences.

`base`:
Causal, Sorted, Test Session restricted to 300 trials (5 minutes). ~26K trials.
- Patch 4, 16, 32, 128
- Stitch
- NDT-1 (Time), 6 and 8 layer (control for capacity).

Batch size controlled at 256, though we don't expect it to matter much.

`scaledown`:
TODO...

`cross`:
- `subject_loco`: Cross animal
- `task_co`: Cross task, using Gallego CO + NLB Maze reaches (multiple, to increase trial count)


# Env creation
For training on cluster systems, please refer to ENV.yml, which should be sufficient.
Not so easy to go cross platform, so consider the following steps to construct from scratch:

(For Windows)
- `conda create -n ndt2 -c conda-forge -y python==3.10.6`
(Pytorch)
- `conda install pytorch==1.13.1 torchvsion==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia`
Other core infra
- `conda install -y -c conda-forge dacite dandi einops hydra-core h5py pytorch-lightning pandas seaborn scikit-learn`
- `pip install wandb`

Various libs will be needed to deal with individual datasts, but those likely aren't needed if you're using a pretrained model..