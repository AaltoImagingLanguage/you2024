#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:34:19 2023
plot: final version of LME with 20 ms time window
fixed effects:
    1) # Replaced letters
    2) euclidean
    3) Difference of bigram frequency
    

@author: youj2
"""

# %%
# %matplotlib qt
import numpy as np
import mne
import os
from mne.minimum_norm import make_inverse_operator, apply_inverse, read_inverse_operator
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd
# import pingouin as pg
import mne
import statsmodels.formula.api as smf
from config import fname, event_id, baseline,subjects, rois_id, parc,rois_names,roi_colors
import utility.figure_setting
import warnings
from utils import select_rois
from scipy.signal import savgol_filter
warnings.filterwarnings('ignore')
from statsmodels.stats.multitest import multipletests
method='fdr_bh'
# %%
step=10
color = plt.get_cmap("tab20b").colors
#Difference of bigram frequency,Accuracy
varible="Accuracy"#,euclidean#'euclidean'#'condition',length,old20,Ratio of replaced vowels,# Replaced letters,euclidean
print(varible)
SUBJECT = 'fsaverage'
mne.set_config('SUBJECTS_DIR', fname.mri_subjects_dir)
hemi='lh'
if hemi=='rh':
    rois_id=[i+1 for i in rois_id]
n_roi=3
size=800
p_val=0.01
length = 651
rois_id=rois_id[:n_roi]
rois_names=rois_names[:n_roi]
roi_colors=roi_colors[:n_roi]
# roi_colors.reverse()
labels=select_rois(rois_id=rois_id,
                   parc=parc,
                    combines=[]
                    # combines=[[0,1]]
                   )

# %%
legend=True
stc = mne.read_source_estimate(
    fname.stc_morph(subject='sub-01', category='RW'))
# stc.crop(0,1.1)
x = stc.times
data = np.load(f'{fname.data_dir}/rois_lmes_tc_{varible}_{step}window.npy')  # (2,len(rois), length)
# data = data[:,:,100:]#from 0 ms 
# data = results
# n_rois = data.shape[1]
# color = 'forestgreen' #'darkseagreen'  # 'cornflowerblue',darkorange
# roi_colors = [ 'r','g','b','k',]

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
# fig.suptitle("Ratio of replaced vowels",fontsize=18,weight='bold')
if varible=='euclidean':
    varible='Visual distance'
if varible=='# Replaced letters':
    varible='No. of replaced letters'
if varible=='Difference of bigram frequency':
    varible='Bigram frequency difference'
if varible=='Accuracy':
    varible='Recognizability'
fig.suptitle(varible,fontsize=18,weight='bold')
ax.spines[['right', 'top']].set_visible(False)

if varible=='Accuracy':
    
    ax.set_ylabel('|β|')#β
else:
    ax.set_ylabel('β')#β
# %%
for i in range(n_roi):
    
    y = np.repeat(data[0, i,:],step)
    y = y[:len(x)]
    y=np.abs(y)
    #smoothing the data
    y=savgol_filter(y, 75,4,
                    # mode='nearest'
                    )
    error = data[1, i, :]
    error = np.repeat( error,step)
    error = error[:len(x)]
    ax.plot(1e3 *x, y,
            color=roi_colors[i],
            label=rois_names[i]
            # linewidth=1.2
            # alpha=0.8
            
            )
    ax.fill_between(1e3 *x, y-error, y+error,
                    alpha=0.1,
                    color=roi_colors[i],
                    linewidth=0)

    # regions of significant effect
    # p_value = data[2, i, :]
    p_value=multipletests(data[-1,i,:],method = method)[1]
    p_value = np.repeat( p_value,step)
    p_value = p_value[:len(x)]
    where = np.where(p_value < p_val)
    # if i == 4:
    #     ax.plot(1e3 *x[where[0]], [-0.17]*len(where[0]), "s",
    #             color=color,
    #             alpha=1,
    #             # label=f'{cat} - RW (p<0.05)'
    #             )
   
    ax.plot(1e3 *x[where[0]], 
            [-0.231+i*0.031]*len(where[0]),#index
                # [-0.5+i*0.07]*len(where[0]),#length
                # [-0.2+i*0.059]*len(where[0]),#vowels_ratio
                # [-3+i*0.5]*len(where[0]),#freq
                # [-0.24+i*0.032]*len(where[0]),#visual distance
                # [-0.2-i*0.025]*len(where[0]),#index
                "s",
            color=roi_colors[i],
            # alpha=0.8,
            # label=f'{cat} - RW (p<0.05)'
            )
if varible in ["No. of replaced letters",]:
    leg =fig.legend(
        # loc='upper left', 
                ncol=1, 
                bbox_to_anchor=(0.38, 0.912)
                )
    for legobj in leg.legend_handles:
        legobj.set_linewidth(6)
    ax.set_xlabel('Time (ms)')
plt.ylim(-0.25, 0.8)
plt.xlim(-200, 1100)
f=f'{fname.figures_dir(subject=SUBJECT)}/rois-lmem_{hemi}'
if not os.path.exists(f):
      os.makedirs(f, exist_ok=True)
plt.savefig(
      f'{f}/all_rois_smoothed_filled_fixed_{varible}_final_{step}window_pcorrect_p{p_val}.pdf',
      bbox_inches='tight'
  )


