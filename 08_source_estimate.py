#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:06:45 2023

@author: jiaxin
"""

# %% import modules
import argparse
import mne
from mne.minimum_norm import apply_inverse, read_inverse_operator
from config import fname, event_id

# %%
# Be verbose
mne.set_log_level('INFO')


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--subject',  type=int, help='The subject to process')
args = parser.parse_args()
print('Processing subject:', args.subject)

subject = f'sub-{args.subject:02}'
log = fname.log(subject=args.subject, proc='stc')
mne.set_log_file(log, overwrite=True)

# %% Read the epochs
print('Reading epochs...')
epochs = mne.read_epochs(fname.epo(subject=subject))

# %% config
mne.set_config('SUBJECTS_DIR', fname.mri_subjects_dir)
ave = {cat: epochs.crop(tmax=1.1)[cat].average() for cat in event_id}

# %%
source_estimates = []
method = 'dSPM'
snr = 3.0
lambda2 = 1. / snr ** 2
inv = read_inverse_operator(fname.inv(subject=subject))
src_to = mne.read_source_spaces(fname.fsaverage_src)

# %%
for i, cat in enumerate(event_id):

    source_estimate = apply_inverse(ave[cat], inv, lambda2,
                                    method=method,
                                    pick_ori=None)

    source_estimate.save(
        fname.stc(subject=subject, category=cat), overwrite=True)
    morph = mne.compute_source_morph(
        inv['src'], subject_from=subject, subject_to='fsaverage',
        src_to=src_to
    )
    morphed_stc = morph.apply(source_estimate)

    # %%
    morphed_stc.save(fname.stc_morph(
        subject=subject, category=cat), overwrite=True)
