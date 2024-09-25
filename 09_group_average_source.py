#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grand avarage stc for each condition

@author: youj2
"""

# %% import modules
from mne.datasets import fetch_fsaverage
import argparse
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse, read_inverse_operator
from config import (fname, event_id, bad_channels, subjects)
import os
import os.path as op
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats as stats
from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
# %%

typ = 'meg'
SUBJECT = 'fsaverage'
mne.set_config('SUBJECTS_DIR', fname.mri_subjects_dir)
# %%
src_to = mne.read_source_spaces(fname.fsaverage_src)
# %%
for i, cat in enumerate(event_id):
    stcs = []
    for sub in subjects:
        src = mne.read_source_spaces(fname.src(subject=sub))
      
	f = fname.stc(subject=sub, category=cat)
	f_morph = fname.stc_morph(subject=sub, category=cat)
	f_save = fname.ga_stc(category=cat)
        
        stc = mne.read_source_estimate(f)
        morph = mne.compute_source_morph(
            src, subject_from=sub, subject_to=SUBJECT,
            src_to=src_to
        )
        stc_morph = morph.apply(stc)
        stcs.append(stc_morph)
        Path(f_morph).parent.mkdir(parents=True, exist_ok=True)
        stc_morph.save(f_morph, overwrite=True)

    ga_stc = np.mean(stcs)
    Path(f_save).parent.mkdir(parents=True, exist_ok=True)
    ga_stc.save(f_save, overwrite=True)
