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
        'torch==1.13.1+cu117', # 2.0 onnx export doesn't work, install with --extra-index-url https://download.pytorch.org/whl/cu117
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
        'pytorch-lightning',
        'scikit-learn',
        'ordered-enum',
        'mat73',
        'dacite',
        'gdown',
    ],
)