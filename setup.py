from setuptools import setup, find_packages

setup(
    name='context_general_bci',
    version='0.0.1',

    url='https://github.com/joel99/context_general_bci',
    author='Joel Ye',
    author_email='joelye9@gmail.com',

    packages=find_packages(exclude=['scripts', 'crc_scripts', 'data']),
    py_modules=['context_general_bci'],
)