# context_general_bci
Codebase for Neural Data Transformer 2.

## Getting started

### Requirements
Most core requirements for model training are listed in `env.yaml`:
```
conda env create -f env.yaml
```
This list was designed with non-linux in mind and requires some more manual steps may be needed i.e. as not all packages are available on conda.
Some additional packages must be installed via pip:
- `pip install ordered-enum mat73 dacite gdown`

Setup this repo with `pip install -e .` (or `pip install .` if you don't plan on editing the codebase). We make several dataloading dependencies optional (but training with said data will fail without them).

Note for slurm jobs, I trigger the necessary env loads with a `load_env.sh` script located outside this repo, samples provided, but you will need to edit them to match your env.

Data setup is not modularized, some good portion can be done with the following command; individual dataloaders in `tasks` have specific instructions for the remainder.
```
. install_datasets.sh
```
Several datasets needs specific data processing libs. Set those up outside this repo:
```
git clone git@github.com:NeuralAnalysis/PyalData.git
cd PyalData
pip install .

git clone git@github.com:neurallatents/nlb_tools.git
cd nlb_tools
pip install .
```
### Running an experiment
Logging is done on wandb, which should be set up before runs are launched.
Provided all paths are setup, start a given run with:
`python run.py +exp/<EXP_SET>=<exp>`
e.g. to run the experiment configured in `context_general_bci/config/exp/arch/base/f32.yaml`: `python run.py +exp/arch/base=f32`.
You can launch on slurm via `sbatch ./launch.sh +exp/<EXPSET>=<exp>`.
A whole folder can be launched through slurm with `python launch_exp.py -e ./context_general_bci/config/exp/arch/base`.

Configurations for hyperparameter sweeping can be configured, see e.g. `exp/arch/tune_hp`.

## Notes
### Codebase design
This codebase mixes many different heterogenuous datasets in an attempt to make more general neural data models. We also assume Transformers as the only model backbone. This is a BERT era effort to get large base models on which various BCI tasks will be solved. Being BERT era, the default task strategy will be fine-tuning.


### Admin
- note that we installed NLB tools via pip and that this constrained our pandas to be <1.34 (whereas it was originally ~1.5). A bit annoying - we should go back and re-add NLB tools dependency at some point.

