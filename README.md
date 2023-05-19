# context_general_bci
Codebase for Neural Data Transformer 2. Should provide everything you need to run experiments on public datasets, i.e. RTT results. Codebase is not reduced to the skeleton needed to operate on public datasets -- so please don't mind any extraneous comments and files related to Pitt data.

## Getting started

### Code Requirements
We recommend setting up with conda on a Linux platform Most core requirements for model training are listed in `env_onnx.yml`.
```
conda env create -f env_onnx.yml
```
`env_onnx_cross` is also available for windows setup, though several dependencies are less specified there and might need to be installed individually.

In the conda env, setup this repo with `pip install -e .` (or `pip install .` if you don't plan on editing the codebase). We make several dataloading dependencies optional (but training with said data will fail without them).

### Data Setup
Data setup is not modularized, some good portion can be done with the following command; individual dataloaders in `tasks` have specific instructions for the remainder.
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



## Other Notes
The codebase was actually developed in Python 3.10 but this release uses 3.9 for compatibility with onnx, if you are so interested. That said -- exact numerical reproduction is not guaranteed. Please file an issue if large discrepancies with reported results arise.


### Codebase design
This codebase mixes many different heterogenuous datasets in an attempt to make more general neural data models. We also assume Transformers as the only model backbone. This is a BERT era effort to get large base models on which various BCI tasks will be solved. Being BERT era, the default task strategy will be fine-tuning.



