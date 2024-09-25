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


from scipy import stats
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from config import fname, event_id, baseline, cmaps4, subjects, rois_id,rois_names,roi_colors,window_length
import utility.figure_setting
from utils import create_grow_ROIs2,create_grow_ROIs1,select_rois
import os


#%%

SUBJECT = 'fsaverage'
mne.set_config('SUBJECTS_DIR', fname.mri_subjects_dir)
hemi='lh'
# rois = [rois[-3]]
#%%
# labels = []
# annotation = mne.read_labels_from_annot(
#     'fsaverage', parc='aparc.a2009s_custom_gyrus_sulcus_1100mm2',)
# rois = [label for label in annotation if 'Unknown' not in label.name]
# labels=[rois[i] for i in rois_id]
if hemi=='rh':
    rois_id=[i+1 for i in rois_id]
labels=select_rois(rois_id=rois_id,
                   parc='aparc.a2009s_custom_gyrus_sulcus_800mm2',
                    combines=[]
                    # combines=[[0,1]]
                   )
# %%

length = 651
src_to = mne.read_source_spaces(fname.fsaverage_src)
X = np.zeros([len(event_id), len(subjects), len(labels), length])

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
        label_ts = mne.extract_label_time_course(stc, labels, src_to,
                                                 # mode='mean_flip',
                                                   # mode='pca_flip',
                                                  

                                                 )  #

        # label_ts = stc_baseline_correction(label_ts, stc, tmin, tmax)

        X[c, i, :, :] = label_ts

    # for i in np.arange(0,len(labels)-5):

#%%
np.save(f'{fname.data_dir}/rois_tcs_{hemi}', X)
# %%
X = np.load(f'{fname.data_dir}/rois_tcs_{hemi}.npy')  # (events,subjects,len(rois), length)
stc = mne.read_source_estimate(fname.ga_stc(category='RW'))
stc.crop(-0.2, 1.1)
t = stc.times

test =  'cluster'

heights=[0.044,0.069,0.08,
         ]

# labels=[labels[0]]

# %%
for i in range(len(labels)):
    
    #
    fig, ax = plt.subplots(1, 1,figsize=(8, 5))
    # remove the right and top line frame
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(rois_names[i],color=roi_colors[i],weight='bold',loc='left')

    X_rw = X[0, :, :, :].copy().mean(0)[i, :]
    X_rw=savgol_filter(X_rw, window_length,4,
                    # mode='nearest'
                    )
    ax.plot(1e3 *t, X_rw, color=cmaps4[0], label=list(event_id.keys())[0])
    
    
    for c in range(len(list(event_id.keys()))-1):
        cat = list(event_id.keys())[c+1]
        X_pw=X[c+1, :, :, :].copy().mean(0)[i, :]
        X_pw=savgol_filter(X_pw, window_length,4,
                        # mode='nearest'
                        )
        ax.plot(1e3 *t, X_pw , color=cmaps4[c+1], 
                label=cat
                )
        
        # not meet the assumptions of normality or homogeneity of variance
        if test == 'wilcoxon':
            p_value=np.zeros(X.shape[-1])
            for tt in range(X.shape[-1]):
                tt_min=tt-radius if tt>radius else tt
                tt_max=tt+radius if tt>radius else tt+2*radius 
                t_values, p = stats.wilcoxon(X[c+1, :, i, tt_min:tt_max].mean(-1),
                                             X[0, :, i, tt_min:tt_max].mean(-1),
                alternative='greater',
                correction=True
                )
                p_value[tt]=p
            where = np.where(p_value < 0.05)
            ax.plot(1e3 *t[where[0]], [-0.05-c*heights[i]]*len(where[0]), "s", color=cmaps4[c+1],
                      alpha=1,
                      # label=f'{cat} - RW (p<0.05)'
                      )
        elif test == 't':
            t_values, p_value = stats.ttest_rel(
                X[c+1, :, i, :], X[0, :, i, :],
                alternative='greater'
            )
        elif test=='cluster':
            t_obs, clusters, pvals, H0 = mne.stats.permutation_cluster_1samp_test(X[c+1,:,i,:]-X[0,:,i,:],
                                                                                   threshold=1.5,
                                                                                  tail=0,
                                                                                  )
            good_clusters_idx = np.where(pvals < 0.05)[0]
            good_clusters = [clusters[idx] for idx in good_clusters_idx]
            print('n_cluster=',len(good_clusters))
            for jj in range(len(good_clusters)):
                ax.plot(1e3 *t[good_clusters[jj]], [-0.05+c*heights[i]]*len(good_clusters[jj][0]), 's', color=cmaps4[c+1],
                                            alpha=1,
                                            # label=cat
                                            )
        
    
    if i==1:
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Source amplitude (dSPM value)')
        leg =fig.legend(
            # loc='best', 
                   ncol=1, 
                    # bbox_to_anchor=(1, 1)
                   )
        # set the linewidth of each legend object
        for legobj in leg.legend_handles:
            legobj.set_linewidth(6)
    # fig.tight_layout()
    f=f'{fname.figures_dir(subject=SUBJECT)}/rois-tc_{hemi}_smoothed_{window_length}_n{len(subjects)}_({test}_test)'
    if not os.path.exists(f):
        os.makedirs(f, exist_ok=True)
    plt.savefig(
        f'{f}/roi{i}_tcs__brain.pdf',
        bbox_inches='tight'
    )
    # plt.savefig(
    #     f'{fname.figures_dir(subject=SUBJECT)}/roi/roi{i}_tcs_({test}_test).pdf',
    #     bbox_inches='tight'
    # )

# %%
