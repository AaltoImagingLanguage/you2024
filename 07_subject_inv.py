#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a source space by for the fsaverage brain.
"""
# %% import modules
import argparse
import mne
from mne.minimum_norm import make_inverse_operator, write_inverse_operator
from config import fname, spacing
# %%
# Be verbose
mne.set_log_level('INFO')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--subject',  type=int, help='The subject to process')
args = parser.parse_args()

# MEG-MRI co-registration??
co_reg = True


print('Processing subject:', args.subject)
subject = f'sub-{args.subject:02}'

# %% Read the epochs
print('Reading epochs...')
epochs = mne.read_epochs(fname.epo(subject=subject))

mne.set_config('SUBJECTS_DIR', fname.mri_subjects_dir)

noise_cov = mne.compute_covariance(
    epochs, tmax=0.0, method='auto', rank='info')
noise_cov = mne.cov.regularize(noise_cov, epochs.info)

# %% Establish the source space
src = mne.setup_source_space(subject=subject, spacing=spacing, add_dist=False)
src.save(fname.src(subject=subject), overwrite=True)


# %% Create the BEM model
model_surfaces = mne.make_bem_model(
    subject=subject, conductivity=[0.3])
bem = mne.make_bem_solution(model_surfaces)


# Compute the lead-field matrix, i.e, the forward solution
fwd = mne.make_forward_solution(
    epochs.info, trans=fname.trans(subject=subject), src=src, bem=bem, n_jobs=-1,
    # mindist=5,
    eeg=False)
mne.write_forward_solution(fname.fwd_r(subject=subject), fwd, overwrite=True)
fwd = mne.convert_forward_solution(fwd, surf_ori=True)
# Use only MEG channels
fwd = mne.pick_types_forward(fwd, meg=True, eeg=False)

inv = make_inverse_operator(epochs.info, fwd, noise_cov,
                            loose=0.2,
                            # loose='auto',
                            depth=0.8)
write_inverse_operator(fname.inv(subject=subject), inv, overwrite=True)
