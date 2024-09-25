#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:06:45 2023

@author: jiaxin
"""

# %% import modules
import argparse
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse, write_inverse_operator, read_inverse_operator
from config import fname, event_id
import os
import os.path as op
import matplotlib.pyplot as plt
from mne.datasets import fetch_fsaverage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

# %%
# Be verbose
mne.set_log_level('INFO')


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--subject',  type=int, default=6,
                    help='The subject to process', required=False)
args = parser.parse_args()
print('Processing subject:', args.subject)
subject = f'sub-{args.subject:02}'
log = fname.log(subject=args.subject, proc='stc')
#
mne.set_log_file(log, overwrite=True)
# %% Read the epochs
print('Reading epochs...')
epochs = mne.read_epochs(
    fname.epo(subject=subject))

# %% config
SUBJECT = subject
mne.set_config('SUBJECTS_DIR', fname.mri_subjects_dir)
# %%
# ave = {cat: epochs[cat].average().pick_types('grad') for cat in event_id}
# mne.viz.evoked.plot_evoked_topo([ave[i] for i in ave.keys()],
#                                 # scalings=dict(grad=3e13, mag=5e14),
#                                 ylim=dict(grad=[-30, 30])
#                                 )
# %% Prepare brain
# epochs.pick_types('grad')
ave = {cat: epochs.crop(tmax=1.1)[cat].average() for cat in event_id}

# %%
source_estimates = []
method = 'dSPM'
snr = 3.0
lambda2 = 1. / snr ** 2
# %%
inv = read_inverse_operator(fname.inv(subject=SUBJECT))
src_to = mne.read_source_spaces(fname.fsaverage_src)
# src = mne.read_source_spaces(fname.src(subject=SUBJECT))
# %%
for i, cat in enumerate(event_id):

    source_estimate = apply_inverse(ave[cat], inv, lambda2,
                                    method=method,
                                    pick_ori=None)

    source_estimate.save(
        fname.stc(subject=SUBJECT, category=cat), overwrite=True)
    morph = mne.compute_source_morph(
        inv['src'], subject_from=SUBJECT, subject_to='fsaverage',
        src_to=src_to
    )
    morphed_stc = morph.apply(source_estimate)
    # print(fname.ga_stc1(subject=SUBJECT, category=cat))

    # %%
    # morphed_stc.save(rsa_f,overwrite=True)
    morphed_stc.save(fname.stc_morph(
        subject=SUBJECT, category=cat), overwrite=True)
