#%%
import os, pickle
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator, DATASET_HELDINOUT_MAP

# argparse the eval set
import sys
import argparse

track = 'h1'
eval_metrics_path = f'/snel/home/bkarpo2/bin/falcon-challenge/manuscript/results/{track}_eval_rnn.csv'
ndt2_run_df = pd.read_csv(eval_metrics_path)

#%%
ndt2_run_df.head()

#%% 
def reduce_dataset_from_variant(variant):
    if 'm2' in variant:
        if 'ses' in variant:
            return variant.split('ses-')[-1].split('.*')[0].replace('-', '')
        else:
            return variant.split('.*_')[1].split('_')[0]
    elif 'h1' in variant:
        return variant.split('FALCONH1-')[1].split('_')[0]
    elif 'm1' in variant:
        if 'held_out_oracle' in variant:
            return variant.split('_held_out_oracle-')[0].split('FALCONM1-L_')[-1]
        else:
            return variant.split('ses-')[-1].split('.*')[0].replace('-', '')
    else:
        raise ValueError(f"Unknown variant {variant}")
def get_own_day_r2(row):
    # datasets are reduced, so we can't use falcon-challenge directly for h1/m2.
    if 'h1' in row.variant:
        if row.dataset in ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']:
            return row[f'Held In {row.dataset} R2']
        elif row.dataset in ['S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12']:
            return row[f'Held Out {row.dataset} R2']
    elif 'm2' in row.variant:
        if row.dataset in ['20201019', '20201020', '20201027', '20201028']:
            return row[f'Held In {row.dataset} R2']
        elif row.dataset in ['20201030', '20201118', '20201119', '20201124']:
            return row[f'Held Out {row.dataset} R2']
    elif 'm1' in row.variant:
        if row.dataset in DATASET_HELDINOUT_MAP['m1']['held_in']:
            return row[f'Held In {row.dataset} R2']
        elif row.dataset in DATASET_HELDINOUT_MAP['m1']['held_out']:
            return row[f'Held Out {row.dataset} R2']
    print(row)
    raise ValueError(f"Unknown dataset {row.dataset}")

ndt2_run_df['dataset'] = ndt2_run_df.variant.apply(reduce_dataset_from_variant)
ndt2_run_df['Heldout_Accuracy'] = ndt2_run_df.apply(get_own_day_r2, axis=1)

#%% 
def grouped_heldout_accuracy(df):
    if 'h1' in df.variant.iloc[0]:
        subset = df[df.dataset.isin(['S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12'])]['Heldout_Accuracy']
    elif 'm2' in df.variant.iloc[0]:
        subset = df[df.dataset.isin(['20201030', '20201118', '20201119', '20201124'])]['Heldout_Accuracy']
    elif 'm1' in df.variant.iloc[0]:
        subset = df[df.dataset.isin(DATASET_HELDINOUT_MAP['m1']['held_out'])]['Heldout_Accuracy']
    return subset.mean(), subset.std()

def grouped_heldin_accuracy(df):
    if 'h1' in df.variant.iloc[0]:
        subset = df[df.dataset.isin(['S0', 'S1', 'S2', 'S3', 'S4', 'S5'])]['Heldout_Accuracy']
    elif 'm2' in df.variant.iloc[0]:
        subset = df[df.dataset.isin(['20201019', '20201020', '20201027', '20201028'])]['Heldout_Accuracy']
    elif 'm1' in df.variant.iloc[0]:
        subset = df[df.dataset.isin(DATASET_HELDINOUT_MAP['m1']['held_in'])]['Heldout_Accuracy']
    return subset.mean(), subset.std()

print(ndt2_run_df[['dataset', 'Heldout_Accuracy']])

print("Max-of-set metrics (every single-day model evaluated on their day)")
print(f'Heldout: {grouped_heldout_accuracy(ndt2_run_df)}')
print(f'Heldin: {grouped_heldin_accuracy(ndt2_run_df)}')
#%% 
oracle_perfs = ndt2_run_df[['dataset', 'Heldout_Accuracy']]

oracle_performance = []
if track == 'h1':
    for i in range(13):
        oracle_performance.append(oracle_perfs.loc[oracle_perfs['dataset'] == f'S{i}']['Heldout_Accuracy'].values)
elif track == 'm1': 
    for i in DATASET_HELDINOUT_MAP['m1']['held_in'] + DATASET_HELDINOUT_MAP['m1']['held_out']: 
        oracle_performance.append(oracle_perfs.loc[oracle_perfs['dataset'] == str(i)]['Heldout_Accuracy'].values)
elif track == 'm2': 
    for i in ['20201019', '20201020', '20201027', '20201028', '20201030', '20201118', '20201119', '20201124']:
        oracle_performance.append(oracle_perfs.loc[oracle_perfs['dataset'] == i]['Heldout_Accuracy'].values)

#%% 
def compute_other_held_in_mean_perf(df):
    if 'h1' in df.variant.iloc[0]:
        subset = df[df.dataset.isin(['S0', 'S1', 'S2', 'S3', 'S4', 'S5'])]
        subset['Other_Accuracy'] = subset.apply(lambda row: sum(
            [row[f'Held In {held_in} R2'] for held_in in ['S0', 'S1', 'S2', 'S3', 'S4', 'S5'] if held_in != row.dataset]) / 5, axis=1)
    elif 'm2' in df.variant.iloc[0]:
        subset = df[df.dataset.isin(['20201019', '20201020', '20201027', '20201028'])]
        subset['Other_Accuracy'] = subset.apply(lambda row: sum(
            [row[f'Held In {held_in} R2'] for held_in in ['20201019', '20201020', '20201027', '20201028'] if held_in != row.dataset]) / 3, axis=1)
    elif 'm1' in df.variant.iloc[0]:
        subset = df[df.dataset.isin(DATASET_HELDINOUT_MAP['m1']['held_in'])]
        subset['Other_Accuracy'] = subset.apply(lambda row: sum(
            [row[f'Held In {held_in} R2'] for held_in in DATASET_HELDINOUT_MAP['m1']['held_in'] if held_in != row.dataset]) / (len(DATASET_HELDINOUT_MAP['m1']['held_in']) - 1), axis=1)
    else:
        raise ValueError(f"Unknown variant {df.variant.iloc[0]}")
    # find the best row
    best_held_in_day = subset.loc[subset.groupby('eval_set')['Other_Accuracy'].idxmax()]
    return best_held_in_day
    # return best_hexld_in_day

static_row = compute_other_held_in_mean_perf(ndt2_run_df)
print(f"Static (choose best day)")
print(f"Static all: {static_row}")

#%% 
print(f"----")
print(f'heldout r2: {static_row["eval_r2"]}')
print(f'heldout r2 std: {static_row["Held Out R2 Std."]}')
print(f'heldin r2: {static_row["heldin_eval_r2"]}')
# %%
static_performance = []
if track == 'h1':
    for i in range(13):
        string = 'Held In' if i <= 5 else 'Held Out'
        static_performance.append(static_row[f'{string} S{i} R2'].values)
elif track == 'm1': 
    for i in DATASET_HELDINOUT_MAP['m1']['held_in'] + DATASET_HELDINOUT_MAP['m1']['held_out']:
        string = 'Held In' if i in DATASET_HELDINOUT_MAP['m1']['held_in'] else 'Held Out'
        static_performance.append(static_row[f'{string} {i} R2'].values)
elif track == 'm2': 
    for i in ['20201019', '20201020', '20201027', '20201028', '20201030', '20201118', '20201119', '20201124']:
        string = 'Held In' if i in ['20201019', '20201020', '20201027', '20201028'] else 'Held Out'
        static_performance.append(static_row[f'{string} {i} R2'].values)
# %%
with open(f'/snel/home/bkarpo2/bin/falcon-challenge/manuscript/results/{track}_rnn_results2.pkl', 'wb') as f:
    pickle.dump({
        'oracle_performance': oracle_performance,
        'static_performance': static_performance,
    }, f)
# %%
