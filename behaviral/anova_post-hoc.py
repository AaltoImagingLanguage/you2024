#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-way repeated measures ANOVA and post-hoc test

@author: youj2
statistical analysis of the behaviroul results
"""
# %%import modules
import sys
sys.path.append("../")
from config import fname
import pandas as pd
import os
import argparse
import numpy as np
import pingouin as pg
from scipy import stats



# %%read accs and RTs from all subjects
result_folder = f'{fname.behavioral_dir}/behave_result'
accs = pd.read_csv(result_folder+'/accs.csv',
                   index_col=0  # select first column as the row index
                   )
rts = pd.read_csv(result_folder+'/rts.csv', index_col=0)

# %%Exclude subject with abnomal acc
# # response device didn't work properlly for sub-04, may also for sub-11 and sub-14 (acc<0.5)
# threshold = 0
# accs = accs[accs['RW'] > threshold]*100
# rts = rts[accs['RW'] > threshold]*1000

# %%averaged performance of all subjects
#Summary of Behavioral Results of the semantic
# Decision Task (Average RTs and Accuracy) as a Function
# of mispelling level
mean_acc = accs.mean(axis=0)
mean_rt = rts.mean(axis=0)
df = pd.concat([mean_acc, mean_rt], axis=1, keys=['Accuracy(%)', 'RT (msec)'])
print('Summary of Behavioral Results (mean):\n\n', df)

mean_acc = accs.std(axis=0)
mean_rt = rts.std(axis=0)
df = pd.concat([mean_acc, mean_rt], axis=1, keys=['Accuracy(%)', 'RT (msec)'])
print('Summary of Behavioral Results (std):\n\n', df)
# %%A one-way repeated measures (i.e., related groups) ANOVA:N0: they are different
# participants have been subjected to more than one condition/trial and the response to each of these conditions is to be compared.
def ow_rm_anova(data):

    res = pg.rm_anova(data)
    df1 = res['ddof1'].iloc[0]
    df2 = res['ddof2'].iloc[0]
    F = res['F'].iloc[0].round(2)
    if res['p-unc'].iloc[0] < 0.001:
        p = 0.001
    elif res['p-unc'].iloc[0] < 0.005:
        p = 0.005
    elif res['p-unc'].iloc[0] < 0.01:
        p = 0.01
    else:
        p = res['p-unc'].iloc[0].round(3)
    result = f'F({df1},{df2})={F}, p<{p}'
    return res, result


res_acc, result_acc = ow_rm_anova(accs)
print(f'acc : {result_acc}')
res_rt, result_rt = ow_rm_anova(rts)
print(f'rt : {result_rt}')
# (F(3,45) = 34.981,p<0.001)

# %% all results
# res_acc, result_acc = ow_rm_anova(accs)
# print(f'acc : {result_acc}')
# res_rt, result_rt = ow_rm_anova(rts)
# print(f'rt : {result_rt}')
# (F(3,45) = 34.981,p<0.001)
# %%post-doc pari-wise test
result_folder = f'{fname.behavioral_dir}/behave_result'
df = pd.read_csv(result_folder+'/performance.csv',
                 )

#%%
# data = pg.read_dataset('rm_anova_wide')
# df = pg.read_dataset('rm_anova')
# anova=pg.rm_anova(dv='rt', within='Type', subject='Subject',
#                                 # padjust='holm',
#                              # alternative='greater',
#                              # parametric='False',
#                              # return_desc=True,
#                              # effsize='cohen',
#                              data=df)
# %% post-hoc pairwise t test
posthocs = pg.pairwise_tests(dv='rt', within='Type', subject='Subject',
                                # padjust='holm',
                                padjust='fdr_bh',#'bonf',
                              # alternative='greater',
                             # parametric='False',
                             # return_desc=True,
                             # effsize='cohen',
                             data=df)
print(posthocs)
# %%accuracy above chance level?
results = stats.ttest_1samp(accs, 50, alternative='greater')
print(results)
# stats.ttest_1samp(a=list(accs['RL1PW']),
#                   popmean=50,
#                   alternative='greater')
# Overall task performance was above chance except for RL3PW
