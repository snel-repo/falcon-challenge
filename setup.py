from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requires = f.read().splitlines()

setup(
    name='falcon_challenge',
    version='0.0.1',

    url='https://github.com/snel-repo/stability-benchmark',
    author='Joel Ye',
    author_email='joelye9@gmail.com',

    packages=find_packages(where='falcon_challenge'),
    requires=[
        'numpy',
        'hydra',
        'matplotlib',
        'tqdm',
        'scipy',
        'pandas',
        'seaborn',
        'scikit-learn',
    ],
)