# to be called from project root
# ln -s ~/share/ndt2/ data # your share dir here, i.e. call from ./data

# assert data exists
if [ ! -d "data" ]; then
  echo "data/ does not exist, please create it e.g. a symlink to your data directory"
  exit 1
fi

cd data
# RTT
mkdir odoherty_rtt
cd odoherty_rtt
zenodo_get 3854034
cd ..

# NLB
mkdir nlb
cd nlb
dandi download DANDI:000128/0.220113.0400
dandi download https://dandiarchive.org/dandiset/000129/draft
dandi download DANDI:000138/0.220113.0407
dandi download DANDI:000139/0.220113.0408
dandi download DANDI:000140/0.220113.0408
cd ..

# TODO use NWB formatted data instead of google drive scrapes (this is just nir-even's data)
# This call requires this project to be in your PYTHONPATH (else the project-local imports fail in-directory)
mkdir churchland_misc
cd ..
python tasks/churchland_misc.py
cd data


mkdir dyer_co
cd dyer_co
wget -O full-chewie-10032013.mat https://github.com/nerdslab/myow-neuro/blob/main/data/mihi-chewie/raw/full-chewie-10032013.mat?raw=true
wget -O full-chewie-12192013.mat https://github.com/nerdslab/myow-neuro/blob/main/data/mihi-chewie/raw/full-chewie-12192013.mat?raw=true
wget -O full-mihi-03032014.mat https://github.com/nerdslab/myow-neuro/blob/main/data/mihi-chewie/raw/full-mihi-03032014.mat?raw=true
wget -O full-mihi-03062014.mat https://github.com/nerdslab/myow-neuro/blob/main/data/mihi-chewie/raw/full-mihi-03062014.mat?raw=true
cd ..

# Gallego (20ms) and Pitt data are not public.

mkdir -p runs # for wandb