


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
from mne.viz import Brain
import matplotlib as mpl
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd
# import pingouin as pg
import mne
from scipy.signal import savgol_filter
# import pingouin as pg
from config import fname, event_id, baseline, cmaps4, subjects, rois_names,roi_colors,window_length,parc
import figure_setting
from utils import create_grow_ROIs2,create_grow_ROIs1,compare_p
from statsmodels.stats.multitest import multipletests
# import pingouin as pg

#%%
plt.rcParams["font.size"] = 14
plt.rcParams["figure.titlesize"] = 14
SUBJECT = 'fsaverage'
mne.set_config('SUBJECTS_DIR', fname.mri_subjects_dir)
hemi='both'
method='fdr_bh'
# rois = [rois[-3]]
#%%
annotation = mne.read_labels_from_annot(
    'fsaverage', parc='aparc.a2009s_custom_gyrus_sulcus_800mm2',)
# annotation = mne.read_labels_from_annot(
#     'fsaverage', parc='aparc.a2009s',)
#%%

rois = [label for label in annotation if 'Unknown' not in label.name]
    
#%%growed lables
# stcs=[]
# mne.set_config('SUBJECTS_DIR', fname.mri_subjects_dir)
# SUBJECT = 'fsaverage'
# for i, cat in enumerate(event_id):
#     stc = mne.read_source_estimate(fname.ga_stc1(category=cat))
#     stcs.append(stc)
#     stc.subject=SUBJECT

# labels=create_grow_ROIs1(stcs, extents=15)
# %%

length = 651
src_to = mne.read_source_spaces(fname.fsaverage_src)
X = np.zeros([len(event_id), len(subjects), len(rois), length])

tmin, tmax = baseline


def stc_baseline_correction(ts, stc, tmin, tmax):
    time_dim = len(stc.times)
    # baseline_timepoints = X.times[np.where(X.times<0)]
    # baseline_timepoints = X.times[np.where(X.times==tmin):np.where(X.times==tmax)]
    # Convert tmin/tmax to sample indices
    tmin, tmax = np.searchsorted(stc.times, [tmin, tmax])

    baseline_mean = ts[:, tmin:tmax].mean(1)

    baseline_mean_mat = np.repeat(baseline_mean.reshape([len(baseline_mean), 1]),
                                  time_dim, axis=1)
    corrected_stc = ts - baseline_mean_mat
    return corrected_stc


# %%
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
        label_ts = mne.extract_label_time_course(stc, rois, src_to,
                                                 # mode='mean_flip',
                                                   # mode='pca_flip',
                                                  

                                                 )  #

        # label_ts = stc_baseline_correction(label_ts, stc, tmin, tmax)

        X[c, i, :, :] = label_ts

    # for i in np.arange(0,len(labels)-5):

#%%
np.save(f'{fname.data_dir}/rois_tcs_{hemi}', X)
#%%
def plot_roi_map(values, rois, subject, subjects_dir, cmap="plasma",hemi='both', alpha=1.0):
    
   
    # cmap = mpl.cm.viridis_r
    # cmap = [cmap(i) for i in np.linspace(0, 1, 4)]
    # cmap =list(plt.get_cmap('tab20c').colors[:4])#for rois color
    cmap = mpl.cm.Blues_r
    cmap = [cmap(i) for i in np.linspace(0, 1, 4)]
    
    dic={0.1:3,0.05:2,0.01:1,0.001:0,}

    brain = Brain(
        subject=subject, subjects_dir=subjects_dir, surf="inflated", hemi=hemi,
        # size=(1200, 600),
        views=['lateral', 'ventral'],
        view_layout='vertical',
        cortex="grey",
        background='white'
        
    )
    
    labels_lh = np.zeros(len(brain.geo["lh"].coords), dtype=int)
    labels_rh = np.zeros(len(brain.geo["rh"].coords), dtype=int)
    ctab_lh = list()
    ctab_rh = list()
    for i, (roi, value) in enumerate(zip(rois, values), 1):
        if roi.hemi == "lh":
            labels = labels_lh
            ctab = ctab_lh
        else:
            labels = labels_rh
            ctab = ctab_rh
        labels[roi.vertices] = i
        
        ctab.append([int(x * 255) for x in cmap[dic[value]][:4]] + [i])
    ctab_lh = np.array(ctab_lh)
    ctab_rh = np.array(ctab_rh)
    
    brain.add_annotation(
        [(labels_lh, ctab_lh), (labels_rh, ctab_rh)], borders=False, alpha=alpha,
        remove_existing=False
    )
    brain.add_annotation(parc,borders=True,color='white', remove_existing=False,alpha=alpha)
    return brain
# %%
X = np.load(f'{fname.data_dir}/rois_tcs_{hemi}.npy')  # (events,subjects,len(rois), length)
tmi,tma=0.3,1.1
stc = mne.read_source_estimate(
    fname.stc_morph(subject='sub-01', category='RW'))
tmin, tmax = np.searchsorted(stc.times, [tmi, tma])
X=X[:,:,:,tmin:tmax]
#%%
sig_rois_index=[]#significant 
cats = list(event_id.keys())
figs, axs = plt.subplots(3, 3, figsize=(15, 9))
cmap = mpl.cm.Blues_r
# cmap = [cmap(i) for i in np.linspace(0, 1, 4)]

bounds = [0, 0.001,0.01,0.05,0.1]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N,)
# cbar_ax = figs.add_axes([0.3, 0.02, 0.4, 0.04])
cbar_ax = figs.add_axes([0.8, 0.55, 0.03, 0.2])
# cbar_ax.set_frame_on(False)
b=figs.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cbar_ax, orientation='vertical',
             label="p value")
b.set_ticks([0.001,0.01,0.05])
b.set_ticklabels(['0.001','0.01','0.05',],)
# plt.savefig(f"{fname.figures_dir(subject=SUBJECT)}/rois_cpt4.pdf",
#             bbox_inches='tight'
#             )
#%%

pvalues_all=[]
for j in range(3):
    for k in range(j,3):
        pvalues=[]
        for r in range(len(rois)):
            t_obs, clusters, pvals, H0 = mne.stats.permutation_cluster_1samp_test(X[k+1,:,r,:]-X[j,:,r,:],
                                                                                  # n_permutations=1,
                                                                                   threshold=3,
                                                                                  tail=1,
                                                                         )
            if pvals.size>0:
                p_value = np.min(pvals)
            else:
                p_value=0.1
            
                # p_value=compare_p(p_value)[0]
           
            pvalues.append(p_value) 
        pvalues_all.append(pvalues)

        #coorect p values
        # pvalues=multipletests(pvalues, method = method)[1]
#%%correct p values for each roi
pvalues_all=np.array(pvalues_all)
pvalues_all=np.array([multipletests(pvalues_all[:,i], method = method)[1] for i in range(pvalues_all.shape[1])])
#%%map the pvaules to each scale
#%%
n=0
for j in range(3):
    for k in range(j,3):
        if j==0:
            axs[k, j].set_ylabel( f'{cats[k+1]}')
        if k==2:
            
            axs[k, j].set_xlabel( f'{cats[j]}') 
        pvalues = pvalues_all[:,n]
        pvalues=[compare_p(p_value)[0] for p_value in pvalues]
        brain = plot_roi_map(
            pvalues, rois, subject=SUBJECT, subjects_dir=fname.mri_subjects_dir,
            
            alpha=0.9,
            hemi='split'
        )
        brain.show_view()
        screenshot = brain.screenshot()
        # crop out the white margins
        nonwhite_pix = (screenshot != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
        im = axs[k, j].imshow(cropped_screenshot)
        n+=1   

for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)   


plt.savefig(f"{fname.figures_dir(subject=SUBJECT)}/rois_cpt_{tmi}_{tma}_{method}_corrected_t3.pdf",
            bbox_inches='tight'
            )


