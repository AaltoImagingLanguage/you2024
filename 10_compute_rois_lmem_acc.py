#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:34:19 2023

@author: youj2
"""

# %%
# %matplotlib qt
import numpy as np
import mne
import os

from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd
# import pingouin as pg
import mne
import statsmodels.formula.api as smf
from config import fname, event_id, baseline,subjects, rois_id, parc,rois_names,roi_colors
import utility.figure_setting
import warnings
from utils import create_grow_ROIs2,select_rois
from scipy.signal import savgol_filter
warnings.filterwarnings('ignore')
# %%
# path to raw data
color = plt.get_cmap("tab20b").colors
varible='Accuracy'#'condition',length,old20,Ratio of replaced vowels,# Replaced letters
print(varible)
SUBJECT = 'fsaverage'
mne.set_config('SUBJECTS_DIR', fname.mri_subjects_dir)
hemi='lh'
n_roi=3
size=800
step=10
rois_id=rois_id[:n_roi]
rois_names=rois_names[:n_roi]
roi_colors=roi_colors[:n_roi]
#%%
pclgs = pd.read_csv('/m/nbe/scratch/flexwordrec/scripts/stimuli/psycholinguistics.csv')


# condition=[pclgs.loc[pclgs['type']=='RW',varible].mean(), 
#                               pclgs.loc[pclgs['type']=='RL1PW',varible].mean(), 
#                               pclgs.loc[pclgs['type']=='RL2PW',varible].mean(),
#                               pclgs.loc[pclgs['type']=='RL3PW',varible].mean()
#                               ]
condition=[0.863739,0.852087,0.678348,0.498522]#mean accs for each condition
# %%
condition=(condition-np.min(condition))/(np.max(condition)-np.min(condition))
# condition=(condition-np.mean(condition))/np.std(condition)
mean_chunks=condition
#%%
# condition=[0.5, 0.9, 0.6,1]
# rois = [rois[3]]
# labels = []
# annotation = mne.read_labels_from_annot(
#     'fsaverage', parc=parc,)
# for roi in rois:
#     label = [label for label in annotation if label.name ==
#               roi+hemi][0]
#     labels.append(label)

labels=select_rois(rois_id=rois_id,
                   parc=parc,
                    combines=[]
                    # combines=[[0,1]]
                   )
# %%
length = 651
src_to = mne.read_source_spaces(fname.fsaverage_src)
X = np.zeros([len(event_id), len(subjects), len(labels), length])

tmin, tmax = baseline
for i in np.arange(0, len(subjects)):
    n_subjects = len(subjects)
    subject = subjects[i]
    print('Participant : ', subject)

    # Average the source estimates within each label using sign-flips to reduce
    # signal cancellations, also here we return a generator
    for c, cat in enumerate(event_id):
        stc = mne.read_source_estimate(
            fname.stc_morph(subject=subject, category=cat))

        stc.crop(-0.2, 1.1)
        label_ts = mne.extract_label_time_course(stc, labels, src_to,
                                                 # mode='mean_flip'
                                                 )  # (6,751)

        # label_ts = stc_baseline_correction(label_ts, morphed_stc, tmin, tmax)

        X[c, i, :, :] = label_ts

np.save(f'{fname.data_dir}/rois_tcs_{varible}', X)
# %%test in a roi
X = np.load(f'{fname.data_dir}/rois_tcs_{varible}.npy')  # (2,len(rois), length)
# %%
results = np.zeros([3, len(labels), int(np.ceil(length/step))])
for i in range(len(labels)):

    
    for j, t in enumerate(range(0,length,step)):
        data=pd.DataFrame()
        for n, sub in enumerate(subjects):
       
            values = X[:, n,i, t:t+step].mean(1)[:len(mean_chunks)]
            rows=pd.DataFrame({'Subject': np.repeat(sub, len(mean_chunks)),
                                'condition': np.tile(mean_chunks, 1),
                                  'Values': values})
            data=pd.concat([data,rows],ignore_index=True)

        md = smf.mixedlm("Values ~ condition", data, groups=data["Subject"],
                          # re_formula="~1+condition",
                          )
        # %https://www.statsmodels.org/dev/optimization.html
        lmm_full = md.fit(
            method='powell'
            )
        # print(lmm_full.summary())

        if not lmm_full.converged:
            # raise ValueError("The model didn't converge")
            std =0
            beta = 0
            p = 1
            results[:, i, j] = [beta, std, p]
        else:
            std = lmm_full.bse_fe['condition']
            # beta = np.abs(lmm_full.params['condition'])
            beta = lmm_full.params['condition']
            p = lmm_full.pvalues['condition']
            results[:, i, j] = [beta, std, p]
        print('roi', i, 't', j, '\n')
        #p values corretion
    # results[-1,i,:]=multipletests(results[-1,i,:],method = method)[1]
# %%
np.save(f'{fname.data_dir}/rois_lmes_tc_{varible}_{step}window', results)


# %%
# results = np.zeros([3, len(labels), length])
# for i in range(len(labels)):
#     X1 = X[:, :, i, :]
#     for t in range(length):
#         values = X1[:, :, t].T.reshape([-1])

#         data = pd.DataFrame({'Subject': np.repeat(subjects, len(condition)),
#                             'condition': np.tile(condition, len(subjects)),
#                              'Values': values})

#         md = smf.mixedlm(f"Values ~ condition", data, groups=data["Subject"],
#                          # re_formula="~1+condition",
#                          )
#         # %https://www.statsmodels.org/dev/optimization.html
#         lmm_full = md.fit(
#             method='powell'
#             )
#         # print(lmm_full.summary())

#         if not lmm_full.converged:
#             raise ValueError("The model didn't converge")
#         else:
#             std = lmm_full.bse_fe['condition']
#             # beta = np.abs(lmm_full.params['condition'])
#             beta = lmm_full.params['condition']
#             p = lmm_full.pvalues['condition']
#             results[:, i, t] = [beta, std, p]
#         print('roi', i, 't', t, '\n')
# # %%
# np.save(f'{fname.data_dir}/rois_lmes_tc_{varible}', results)
# %%
# legend=True
# x = stc.times
# data = np.load(f'{fname.data_dir}/rois_lmes_tc_{varible}.npy')  # (2,len(rois), length)
# # data = results
# n_rois = data.shape[1]
# # color = 'forestgreen' #'darkseagreen'  # 'cornflowerblue',darkorange
# # roi_colors = [ 'r','g','b','k',]

# fig, ax = plt.subplots(1, 1, figsize=(8, 5))
# # fig.suptitle("Ratio of replaced vowels",fontsize=18,weight='bold')
# fig.suptitle(varible,fontsize=18,weight='bold')
# ax.spines[['right', 'top']].set_visible(False)
# ax.set_xlabel('Time (ms)')
# ax.set_ylabel('β')#β
# # %%
# for i in range(n_rois):
    
#     y = data[0, i, :]
#     y=np.abs(y)
#     #smoothing the data
#     y=savgol_filter(y, 75,4,
#                     # mode='nearest'
#                     )
#     error = data[1, i, :]

#     ax.plot(1e3 *x, y,
#             color=roi_colors[i],
#             label=rois_names[i]
#             # linewidth=1.2
#             # alpha=0.8
            
#             )
#     ax.fill_between(1e3 *x, y-error, y+error,
#                     alpha=0.1,
#                     color=roi_colors[i],
#                     linewidth=0)

#     # regions of significant effect
#     p_value = data[2, i, :]
#     where = np.where(p_value < 0.05)
#     # if i == 4:
#     #     ax.plot(1e3 *x[where[0]], [-0.17]*len(where[0]), "s",
#     #             color=color,
#     #             alpha=1,
#     #             # label=f'{cat} - RW (p<0.05)'
#     #             )
   
#     ax.plot(1e3 *x[where[0]], 
#             [-0.06+i*0.011]*len(where[0]),#index
#                 # [-0.5+i*0.07]*len(where[0]),#length
#                 # [-0.2+i*0.059]*len(where[0]),#vowels_ratio
#                 # [-3+i*0.5]*len(where[0]),#freq
#                 # [-4+i*0.67]*len(where[0]),#visual distance
#                 # [-0.006+i*0.0009]*len(where[0]),#index
#                 "s",
#             color=roi_colors[i],
#             # alpha=0.8,
#             # label=f'{cat} - RW (p<0.05)'
#             )
# if legend:
#     leg =fig.legend(
#         # loc='upper left', 
#                ncol=1, 
#                 bbox_to_anchor=(0.34, 0.912)
#                )
#     for legobj in leg.legend_handles:
#         legobj.set_linewidth(6)
# f=f'{fname.figures_dir(subject=SUBJECT)}/rois-lmem_{hemi}_n{len(subjects)}_size{size}_ave1'
# if not os.path.exists(f):
#      os.makedirs(f, exist_ok=True)
# plt.savefig(
#      f'{f}/all_rois_smoothed_filled_fixed_{varible}_nor.pdf',
#      bbox_inches='tight'
#  )

# print(condition)
