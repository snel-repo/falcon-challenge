from setuptools import setup, find_packages

setup(
    name='falcon_challenge',
    version='0.2.8',

    url='https://github.com/snel-repo/stability-benchmark',
    author='Joel Ye',
    author_email='joelye9@gmail.com',

    packages=find_packages(exclude=['data_demos', 'data']),
    install_requires=[
        'numpy',
        'hydra-core',
        'matplotlib',
        'tqdm',
        'scipy',
        'pandas',
        'seaborn',
        'scikit-learn',
        'pynwb',
        'torch',
        'edit_distance',
        'dandi',
    ],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
