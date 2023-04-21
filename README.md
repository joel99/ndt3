# context_general_bci
Towards large neural data models.

The entire project is structured with the assumption that Transformers will be the backbone.
This is a BERT era effort to get large base models on which various BCI tasks will be solved. Being BERT era, the default task strategy will be fine-tuning.
- we can potentially play with task tokens and task-specific heads for joint training, but there won’t be any mixed-task batches; only text 2 text supports that.
  - [ ]  Check out if any unified vision works manage mixed-task batches?


- For example, to get online decoding; we would need a fine-tuned decoder interface that could run causally.

## Codebase design
This codebase mixes many different heterogenuous datasets in an attempt to make more general neural data models. To keep track of the mess of interfaces, design tends to be strongly typed.

## Admin
- note that we installed NLB tools via pip and that this constrained our pandas to be <1.34 (whereas it was originally ~1.5). A bit annoying - we should go back and re-add NLB tools dependency at some point.


## coding tasklog
- currently no data throw on tokenized path for if a dataset has more channels than we can embed; expect cuda selection errors in that case.


## Requirements
Env per `env.yaml`:
```
conda env create -f env.yaml
```
(Assumes conda is installed, and any necessary cuda packages are available, we use cuda 11.6).
If non-linux, more manual steps may be needed i.e. as not all packages are available on conda.
We make several dataloading dependencies optional (but training with said data will fail without them).
However some packages must be installed via pip:
- `pip install ordered-enum mat73 dacite`
-



Note for slurm jobs, I trigger the necessary env loads with a `load_env.sh` script located outside this repo, samples provided, but you will need to edit them to match your env.

Data setup is not modularized yet, bulk setup is done with:
```
. install_datasets.sh
```
Other things needed:
- pip install mat73
- pip install gdown


Data processing libs will unfortunately also be needed (for now). Call the following outside this repo:
```
git clone git@github.com:NeuralAnalysis/PyalData.git
cd PyalData
pip install .

git clone git@github.com:neurallatents/nlb_tools.git
cd nlb_tools
pip install .
```
