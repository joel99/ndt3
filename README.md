# context_general_bci
Codebase for Neural Data Transformer 2. Should provide everything you need to run experiments on public datasets, i.e. RTT results. Codebase is not reduced to the skeleton needed to operate on public datasets -- so please don't mind any extraneous comments and files related to Pitt data.

## Getting started

### Code Requirements
We recommend setting up with setup.py (the env.ymls lists a dump of an active environment, but setup.py lists the core dependencies)
```
conda create --name onnx python=3.9
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu117
```

In the conda env, setup this repo with `pip install -e .` (or `pip install .` if you don't plan on editing the codebase). We make several dataloading dependencies optional (but training with said data will fail without them).

### Data Setup
Datasets and checkpoints are expected to go under `./data`, please create or symlink that.

Install the public datasets with the following command; for troubleshooting ,individual dataloaders in `tasks` have specific instructions.
```
. install_datasets.sh
```
Several datasets needs specific data processing libs, which can be done with `pip install -r additional_requirements.txt`.


### Running an experiment
Logging is done on wandb, which should be set up before runs are launched (please follow wandb setup guidelines and configure your user in `config_base`.)
Provided all paths are setup, start a given run with:
`python run.py +exp/<EXP_SET>=<exp>`.
e.g. to run the experiment configured in `context_general_bci/config/exp/arch/base/f32.yaml`: `python run.py +exp/arch/base=f32`.

You can launch on slurm via `sbatch ./launch.sh +exp/<EXPSET>=<exp>`, or any of the `launch` scripts. The directives should be updated accordingly. Please note there are several config level mechanisms (`inherit_exp`, `inherit_tag`) in place to support experiment loading inheritance, that is tightly coupled to the wandb checkpoint logging system.
A whole folder can be launched through slurm with `python launch_exp.py -e ./context_general_bci/config/exp/arch/base`.
Note for slurm jobs, I trigger the necessary env loads with a `load_env.sh` script located _outside_ this repo, which then point back into the samples provided (`load_env, load_env_crc.sh`), feel free to edit these to match your local environment needs.


Configurations for hyperparameter sweeping can be configured, see e.g. `exp/arch/tune_hp`. Only manual grid or random searches are currently implemented.

## Checkpoints
Given the relatively lightweight training process we recommend training your own model to be able to use the analysis scripts, which require wandb references. We provide two checkpoints for reference, but note analysis scripts aren't build around manual checkpoint loading.
(Getting a demo analysis is a work in progress.)
Two example checkpoints are provided:
- one from [task scaling](https://wandb.ai/joelye9/context_general_bci/runs/ydv48n02?workspace=user-joelye9): [Checkpoint](https://drive.google.com/file/d/18UgglFKPu6ev5Db4xDtj7aOfzAX4aZy1/view?usp=share_link)
- one from [Indy multisession RTT](https://wandb.ai/joelye9/context_general_bci/runs/uych1wae?workspace=user-joelye9): [Checkpoint](https://drive.google.com/file/d/1hhC4n1UyiYjCcv1nlO6ESljNhr8qVlUF/view?usp=share_link).

## Other Notes
- The codebase was actually developed in Python 3.10 but this release uses 3.9 for compatibility with onnx, if you are so interested. That said -- exact numerical reproduction is not guaranteed. Please file an issue if large discrepancies with reported results arise.
- Check out ./scripts/figures/` and this [wandb workspace](https://wandb.ai/joelye9/context_general_bci) to see how the results were generated.
- This codebase mixes many different heterogenuous datasets in an attempt to make more general neural data models. We also assume Transformers as the only model backbone. This is a BERT era effort to get large base models on which various BCI tasks will be solved. Being BERT era, the default task strategy will be fine-tuning.



