from setuptools import setup, find_packages

setup(
    name='context_general_bci',
    version='0.0.1',

    url='https://github.com/joel99/context_general_bci',
    author='Joel Ye',
    author_email='joelye9@gmail.com',

    packages=find_packages(exclude=['scripts', 'crc_scripts', 'data']),
    py_modules=['context_general_bci'],

    install_requires=[
        'torch==2.1.0+cu118', # 2.0 onnx export doesn't work, install with --extra-index-url https://download.pytorch.org/whl/cu117
        'seaborn',
        'pandas',
        'numpy',
        'scipy',
        'onnxruntime-gpu',
        'pyrtma',
        'hydra-core',
        'yacs',
        'pynwb',
        'argparse',
        'wandb',
        'einops',
        'lightning',
        'scikit-learn',
        'ordered-enum',
        'mat73',
        'dacite',
        'gdown',
        'pyrtma', # For realtime Pitt infra
        'transformers', # Flash Attn
        'packaging', # Flash Attn https://github.com/Dao-AILab/flash-attention
        'rotary-embedding-torch', # https://github.com/lucidrains/rotary-embedding-torch
        'sentencepiece', # Flash Attn
        # 'flash-attn', # install following build instructions on https://github.com/Dao-AILab/flash-attention
        # Add nvcc corresponding to torch (module system on cluster, cuda/11.8)
        # -- export CUDA_HOME=/ihome/crc/install/cuda/11.8
        # pip install flash-attn --no-build-isolation
    ],
)