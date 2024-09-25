#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a source space by for the fsaverage brain.
"""
# %% import modules
import argparse
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse, write_inverse_operator, read_inverse_operator
from config import fname, event_id, spacing
import os
import os.path as op
import matplotlib.pyplot as plt
from mne.datasets import fetch_fsaverage
from mpl_toolkits.axes_grid1 import make_axes_locatable
# %%
# Be verbose
mne.set_log_level('INFO')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--subject',  type=int, default=1,
                    help='The subject to process', required=False)
args = parser.parse_args()

# MEG-MRI co-registration??
co_reg = True


print('Processing subject:', args.subject)
subject = f'sub-{args.subject:02}'

# %% Read the epochs
print('Reading epochs...')
epochs = mne.read_epochs(
    fname.epo(subject=subject))

# %% Prepare brain
# epochs.pick_types('grad')
# ave = {cat: epochs[cat].average() for cat in event_id}
# %% config
SUBJECT = subject
mne.set_config('SUBJECTS_DIR', fname.mri_subjects_dir)
# %%
# MEG-MRI co-registration
# if co_reg:
# mne.gui.coregistration(subject=SUBJECT, inst=fname.epo(
#     subject=subject),
#     # trans=transname
# )
# %%
# empty-room covariance, which captures noise from the sensors and environment.
noise_cov = mne.compute_covariance(
    epochs, tmax=0.0, method='auto', rank='info')
noise_cov = mne.cov.regularize(noise_cov, epochs.info)
# noise_cov.plot(epochs.info)
# %%
# noise_cov = mne.compute_covariance(epochs, method='shrunk', rank='info')
# noise_cov = mne.cov.regularize(noise_cov, epochs.info)
# cov.plot()

# %%
# # Create the BEM model using the FreeSurfer watershed algorithm
# bem=mne.bem.make_watershed_bem(
#     subject=SUBJECT, atlas=True
#     )


# %% Establish the source space

src = mne.setup_source_space(subject=SUBJECT, spacing=spacing, add_dist=False)
src.save(fname.src(subject=SUBJECT), overwrite=True)


# %% Create the BEM model
model_surfaces = mne.make_bem_model(
    subject=SUBJECT, conductivity=[0.3])
bem = mne.make_bem_solution(model_surfaces)


# Compute the lead-field matrix, i.e, the forward solution


# if os.path.isfile(fwdname):
#     fwd = mne.read_forward_solution(fwdname)
# else:
fwd = mne.make_forward_solution(
    epochs.info, trans=fname.trans(subject=SUBJECT), src=src, bem=bem, n_jobs=-1,
    # mindist=5,
    eeg=False)
mne.write_forward_solution(fname.fwd_r(subject=SUBJECT), fwd, overwrite=True)
fwd = mne.convert_forward_solution(fwd, surf_ori=True)
# Use only MEG channels
fwd = mne.pick_types_forward(fwd, meg=True, eeg=False)

# # To avoid re-computing fwd every time, you can d

# %%
inv = make_inverse_operator(epochs.info, fwd, noise_cov,
                            loose=0.2,
                            # loose='auto',
                            depth=0.8)
write_inverse_operator(fname.inv(subject=SUBJECT), inv, overwrite=True)

# %%
# mne.viz.plot_alignment(
#     info=epochs.info,
#     trans=fname.trans(subject=SUBJECT),
#     subject=SUBJECT,
#     src=src,
#     # subjects_dir=subjects_dir,
#     # dig=True,
#     # surfaces=["head-dense", "white"],
#     # coord_frame="meg",
# )
# surfaces = dict(brain=0.4, outer_skull=0.6, head=None)
# mne.viz.plot_alignment(info=epochs.info, trans=transname,
#                        subject=SUBJECT, surfaces=surfaces,
#                        fwd=fwd, src=src,
#                        mri_fiducials=True,
#                        bem=bem)
