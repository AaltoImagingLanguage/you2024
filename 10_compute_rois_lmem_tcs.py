#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:34:19 2023

final_version for time-window LME

@author: youj2
"""

# %%
# %matplotlib qt
import numpy as np
import mne
import os
import argparse
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
from utils import create_grow_ROIs2,select_rois
from scipy.signal import savgol_filter
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
# %%
#'condition',length,old20,Ratio of replaced vowels,# Replaced letters,#Ratio of replaced consonants
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--varible',  type=str, default='# Replaced letters',
                    help='Which predictor for LME', required=False)
parser.add_argument('--max_num',  type=int, default=3,
                    help='How many predictors', required=False)


args = parser.parse_args()
varible = args.varible
n=args.max_num
step=10
method='fdr_bh'
#%%
# path to raw data
color = plt.get_cmap("tab20b").colors
# varible='Bigram frequency'
print(varible)
normalize=True
compute_tcs=True
SUBJECT = 'fsaverage'
mne.set_config('SUBJECTS_DIR', fname.mri_subjects_dir)
hemi='lh'
n_roi=3
size=800
rois_id=rois_id[:n_roi]
rois_names=rois_names[:n_roi]
roi_colors=roi_colors[:n_roi]

if hemi=='rh':
    rois_id=[i+1 for i in rois_id]

labels=select_rois(rois_id=rois_id,
                   parc=parc,
                    combines=[]
                    # combines=[[0,1]]
                   )

#%%sub-13 has the whoele stimuli (540)
epochs = mne.read_epochs(
    fname.epo(subject='sub-13'),verbose=False)
condition=epochs.metadata[varible].tolist()
# metadata = pd.read_csv(f'{fname.data_dir}/stimuli_metadata.csv')
# metadata = metadata[metadata['target'] == '0']  # exclude catch trials
# condition2 = metadata[varible].tolist()
#%%Scatter plot
# r2=metadata['# Replaced letters']
# r1=metadata['euclidean']
# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# plt.show()

#%%
uni_con=np.unique(condition)
condition_chunks=[]
if len(uni_con)>n+1:
    if varible in ['euclidean','Difference of bigram frequency','Difference of letter frequency']:
        condition_chunks=[np.array([0])] 
        uni_con=np.setdiff1d(uni_con,0)#remove 0
    elif varible in ['Ratio of replaced consonants','Ratio of replaced consonants']:
        condition_chunks=[np.array([0])] 
        
        
        
    size=len(uni_con)//n
    left=len(uni_con)%n
    #[3, 2, 2, 6, 2, 45, 5, 9, 12, 13]->[array([2, 3]), array([3, 5]), array([5, 6]), array([6]), array([9]), array([12]), array([13]), array([45])]
    # condition_chunks.extend([uni_con[i:i+size] if i<(n-1)*size else uni_con[i:i+left+size] for i in range(0, len(uni_con)-left, size)])       # condition_chunks=[uni_con[j:j+size] for j in range(0, len(uni_con), size)]
    condition_chunks.extend([uni_con[i*size+i:(i+1)*size+1+i] if i<left else uni_con[(i*size)+left:(i+1)*size+left] for i in range(0, n)])       # condition_chunks=[uni_con[j:j+size] for j in range(0, len(uni_con), size)]


else:
    condition_chunks.extend([np.array([c]) for c in uni_con])

mean_chunks=[chunk.mean() for chunk in condition_chunks]
if normalize:
    mean_chunks=(mean_chunks-np.min(mean_chunks))/(np.max(mean_chunks)-np.min(mean_chunks))
print("condition_chunks",condition_chunks)
print("condition_chunks",mean_chunks)
# %%

length = 651
src_to = mne.read_source_spaces(fname.fsaverage_src,verbose=False)
X = np.zeros([n+3, len(subjects), len(labels), length])#540: num of epochs
conditions={}
tmin, tmax = baseline
#%%
if compute_tcs:
    for i in np.arange(len(subjects)):
        n_subjects = len(subjects)
        subject = subjects[i]
        inv = read_inverse_operator(fname.inv(subject=subject),verbose=False)
        print('Participant : ', subject)
        morph = mne.compute_source_morph(
            inv['src'], subject_from=subject, subject_to='fsaverage',
            src_to=src_to,
            verbose=False
        )
    
        epochs = mne.read_epochs(
            fname.epo(subject=subject),verbose=False)
    
   
        for j, chunk in enumerate(condition_chunks):
            if j==0 and varible in ['euclidean', 'Ratio of replaced vowels','Ratio of replaced consonants']:
                epos=epochs['RW']
                epochs.drop(epochs.metadata['type']=='RW',verbose=False)
                print("len(indices)",len(epos))
            else:
                condition1=epochs.metadata[varible].tolist()
                indices = [i for i, x in enumerate(condition1) if x in chunk]
                epos=epochs[indices]
                print("len(indices)",len(indices))
        
            epochs_stc = apply_inverse(
                epos.average(), inv, lambda2=0.1111,verbose=False)
           
            
            # epochs_stc.crop(0,1.1)    
            morphed_stc = morph.apply(epochs_stc, verbose=False)
                
            label_ts = mne.extract_label_time_course(morphed_stc, labels, src_to,
                                                     verbose=False
                                                      # mode='mean_flip'
                                                      )  # (6,751)
        
            # label_ts = stc_baseline_correction(label_ts, morphed_stc, tmin, tmax)
        
            X[j, i, :, :] = label_ts
    
    

    np.save(f'{fname.data_dir}/rois_tcs_{varible}', X)

#%%
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
