#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:24:41 2024

@author: youj2
statistical analysis of the behaviroul results
"""
# %%import modules
import pandas as pd
import argparse
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from stat_lmm_utils import convert_pvalue_to_asterisks
import statannot
from config import fname,cmaps4
from utility import figure_setting
import os
import sys
import stat_lmm_utils
sys.path.append("../")

plt.rcParams["font.size"] = 14
# %%read accs and RTs from all subjects
result_folder = f'{fname.behavioral_dir}/behave_result'
new_file=False
#%%
if new_file:
    accs = pd.read_csv(result_folder+'/accs_final.csv',
                       index_col=0  # select first column as the row index
                       )
    rts = pd.read_csv(result_folder+'/rts_final.csv', index_col=0)
    perfs = pd.read_csv(result_folder+'/perfs_final.csv', index_col=0)
    perf = {'Subject': [], 'condition': [],
                 'Type': [], 'accuracy': [], 'rt': [],
                 'miss': []
                 }
    for sub in accs.index.values:
        perf['Subject'].extend([sub]*4)
        # perf['condition'].extend(list(accs.columns))
        perf['condition'].extend([0, 1, 2, 3])
        perf['Type'].extend(['RW', 'RL1PW', 'RL2PW', 'RL3PW'])
        perf['accuracy'].extend(list(accs.loc[sub]))
        perf['rt'].extend(list(rts.loc[sub]))
        per = list(perfs.loc[sub])
        miss = [s.split(',')[-1] for s in per]
        perf['miss'].extend(miss)

    df = pd.DataFrame(perf)
    df.to_csv(f'{result_folder}/performance.csv')
else:
    perf = pd.read_csv(result_folder+'/performance.csv',
                       )
# %%
# response device didn't work properlly for sub-04, may also for sub-11 and sub-14 (acc<0.5)
# plt.figure(
#     figsize=(6, 6), dpi=200
#     )
# ax = sns.violinplot(x="Type", y=y, data=perf)
# ax = sns.stripplot(x="Type", y=y, data=perf, jitter=True,
#                     color="white"
#                    )
# x_values = perf["Type"].unique()
# #%%
# statannot.add_stat_annotation(
#     ax,
#     # plot='barplot',
#     data=perf,
#     x="Type",
#     y=y,
#     box_pairs=[(('x','RW'), ('x','RL1PW')), (('x','RL1PW'), ('x','RL2PW')),( ('x','RL2PW'),('x','RL3PW'))],
#     test="t-test_paired",
#     # test='Wilcoxon',
#     text_format="star",
#     line_height=0.02, text_offset=0.1,
#     color='0.2', linewidth=1,
#     # stats_params={'alternative':}
#     # loc="outside",
# )
# %%
fig, axs = plt.subplots(1, 2,
                        figsize=(8,3), 
    tight_layout=False,
    sharey=False) 
ys = ["accuracy", 'rt']

perf['x'] = 'x'
for i, y in enumerate(ys):

    if y == 'rt':
        box_pairs = [(('x', 'RW'), ('x', 'RL1PW')), (('x', 'RL1PW'), ('x', 'RL2PW')),
                     (('x', 'RL2PW'), ('x', 'RL3PW')),
                     (('x', 'RW'), ('x', 'RL2PW'))]
    else:
        box_pairs = [(('x', 'RW'), ('x', 'RL1PW')), (('x', 'RL1PW'), ('x', 'RL2PW')),
                     (('x', 'RL2PW'), ('x', 'RL3PW')),
                     (('x', 'RL1PW'), ('x', 'RL3PW'))]
    alternative = 'less' if y == 'rt' else 'greater'
    sns.boxplot(
        palette=cmaps4,
        fill=False,
        x="x",
        # y=y,
        hue="Type",
        # data=perf, 
        # x="Type", 
        legend=False if i==0 else True,
                   y=y, 
                   # hue="type",
                   data=perf,ax=axs[i],
                    # color='0.2'
    )
    sns.stripplot(
        x='x',
                  y=y,
                    hue='Type', 
                  data=perf,
                    jitter=True,
                    dodge=True,
                   size=4,
                    alpha=0.5,
                    color="black",
                    legend=False,
                  ax=axs[i],
                  # palette=sns.color_palette(),
                  )
  

    # statannot.add_stat_annotation(
    #     axs[i],
    #     # plot='barplot',
    #     data=perf,
    #     x="x",
    #     y=y,
    #     hue="Type",
    #     box_pairs=box_pairs,
    #     test="t-test_paired",
    #     # test='Wilcoxon',
    #     text_format="star",
    #     line_height=0.02, text_offset=0.1,
    #     color='0.2',
    #     # linewidth=2,
    #     stats_params={'alternative': alternative}
    #     # loc="outside",
    # )
    axs[i].spines[['right', 'top',]].set_visible(
        False)  # remove the right and top line frame

    # axs[i].set_xticks(rotation=45)  # remove xticks
    # axs[i].set_xlabel('Type')
    axs[i].set(xticklabels=[])
    axs[i].xaxis.label.set_visible(False)  # remove xaxis label
axs[0].set_ylabel('Accuracy (%)')
axs[1].set_ylabel('Reaction Time (ms)')
# axs[0].get_legend().remove()
handles, labels = axs[1].get_legend_handles_labels()
l = plt.legend(handles, labels, bbox_to_anchor=(
    1.02, 1), loc=2, frameon=False, 
    # fontsize=7
    )
# %%save the figure
SUBJECT = 'fsaverage'
if not os.path.exists(fname.figures_dir(subject=SUBJECT)):
    os.makedirs(fname.figures_dir(subject=SUBJECT))
plt.savefig(f'{fname.figures_dir(subject=SUBJECT)}/acc_rt_statis.png',
            bbox_inches='tight'
            )
plt.savefig(f'{fname.figures_dir(subject=SUBJECT)}/acc_rt_statis.pdf',
            # bbox_inches='tight'
            )
# # https://github.com/webermarcolivier/statannot/blob/master/example/example.ipynb
