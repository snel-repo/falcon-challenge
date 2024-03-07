from setuptools import setup, find_packages

setup(
    name='falcon_challenge',
    version='0.0.1',

    url='https://github.com/snel-repo/stability-benchmark',
    author='Joel Ye',
    author_email='joelye9@gmail.com',

    packages=find_packages(include='falcon_challenge'),
    install_requires=[
        'numpy',
        'hydra-core',
        'matplotlib',
        'tqdm',
        'scipy',
        'pandas',
        'seaborn',
        'scikit-learn'
    ],
)