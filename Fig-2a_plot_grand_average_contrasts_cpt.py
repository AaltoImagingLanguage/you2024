#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:49:16 2023

@author: youj2
"""

# %% import modules
from mne.stats import summarize_clusters_stc
from mne.datasets import fetch_fsaverage
import argparse
import mne
from mne import read_source_estimate
from mne.minimum_norm import make_inverse_operator, apply_inverse, read_inverse_operator
from config import (fname, event_id, subjects)
import os
import os.path as op
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats as stats
from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mne import spatial_src_adjacency
import figure_setting
from plot_cluster import plot_cluster,plot_cluster1
# %% Be verbose
# %% Be verbose
# mne.set_log_level('INFO')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tail',  type=int, default=0,
                    help='two-sided or one-sided paired test', required=False)
parser.add_argument('--pval',  type=float, default=0.05, required=False)
parser.add_argument('--tstep',  type=float, default=0.2, required=False)
parser.add_argument('--tstart',  type=float, default=0.3, required=False)
args = parser.parse_args()

# %%
SUBJECT = 'fsaverage'
# %% Globle parameters for plotting
plt.rcParams["font.size"] = 14
plt.rcParams["figure.titlesize"] = 14
tail = args.tail

p_threshold = args.pval
n_windows =4
threshold=3
hemi = 'split'
n_subjects = len(subjects)

src = mne.read_source_spaces(fname.fsaverage_src)
adjacency = mne.spatial_src_adjacency(src)
print('adjacency.shape:', adjacency.shape)


cats = list(event_id.keys())[1:]
figs, ax = plt.subplots(3, n_windows, figsize=(n_windows*5, 9))
# %%
# clim = dict(kind='value', pos_lims=[0.5, 1, 1.5])
clim = dict(kind='value', lims=[0.1, 0.8, 1.5])

tmins = [round(args.tstart+args.tstep*i, 1) for i in range(n_windows)]
# figs.suptitle('T-maps')
#
# figs.subplots_adjust(right=0.8)
cbar_ax = figs.add_axes([0.925, 0.25, 0.01, 0.5])
# figs.tight_layout()
cbar = mne.viz.plot_brain_colorbar(cbar_ax, clim,
                                   # colormap,
                                   orientation='vertical',
                                   label="F value (dSPM)")
# %%
for j, tmin in enumerate(tmins):
    tmax = round(tmin+args.tstep, 1)  # seconds
    stcs_RW = [read_source_estimate(
        fname.stc_morph(subject=subject, category='RW')).crop(tmin, tmax) for subject in subjects]
    for i, category in enumerate(cats):
        stcs_PW = [read_source_estimate(
            fname.stc_morph(subject=subject, category=category)).crop(tmin, tmax) for subject in subjects]
        
        X = np.concatenate([(stcs_PW[n].data - stcs_RW[n].data)
                            [None, :, :] for n in range(n_subjects)], 0)
        X = X.transpose(0, 2, 1)  # (observations × time × space)
        
        
        # combine all subjects' stc
        stc=np.mean(stcs_PW)-np.mean(stcs_RW)
       
        # %%
        brain = stc.mean().plot(SUBJECT,  hemi=hemi,
                                      views=['lateral', 'ventral'],
                                      clim=clim,
                                      time_label="F value (dSPM)",
                                      time_viewer=False,
                                      background="w",
                                      colorbar=False,
                                      size=(1700, 800)
                                      )
        t_obs, clusters, pvals, H0 = clu = spatio_temporal_cluster_1samp_test(
            X,
            adjacency=adjacency,
            n_jobs=-1,
            seed=1,
            threshold=threshold,

            tail=0
        )   
        good_clusters_idx = np.where(pvals < p_threshold)[0]
        good_clusters = [clusters[idx] for idx in good_clusters_idx]
        for cluster in good_clusters:
            plot_cluster(cluster, src, brain,color="white", width=3)
        # The final part of the cluster permutation test is that you use the smallest cluster p-value as final outcome of the test
        brain.show_view()
        # %%
        if i == 0:
            tmin1000=int(tmin*1000)
            tmax1000=int(tmax*1000)
            ax[i, j].set_title(f'{tmin1000}-{tmax1000} ms\n')

        if j == 0:
            ax[i, j].set_ylabel(f'{category} - RW\n')

        screenshot = brain.screenshot()
        # crop out the white margins
        nonwhite_pix = (screenshot != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
        im = ax[i, j].imshow(cropped_screenshot)
        brain.close()

        ax[i, j].set_yticks([])
        ax[i, j].set_xticks([])
        ax[i, j].set_frame_on(False)


# %%
# We could save this via the following:
if not os.path.exists(fname.figures_dir(subject=SUBJECT)):
    os.makedirs(fname.figures_dir(subject=SUBJECT))
plt.savefig(f'{fname.figures_dir(subject=SUBJECT)}/grand_average_cpt.pdf',
            bbox_inches='tight'
            )
