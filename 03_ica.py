#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Use ICA to remove ECG and EOG artifacts from the data.
"""
# %% import modules
import argparse
import numpy as np
import mne
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs, find_ecg_events, find_eog_events
from mne_bids import BIDSPath, read_raw_bids
from config import (fname, n_ecg_components,
                    n_eog_components, bad_channels, task, eog_chs)
from mne.io.pick import _picks_to_idx, pick_types, pick_channels
from mne.epochs import Epochs
# %%
# Be verbose
mne.set_log_level('INFO')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--subject',  type=int, default=15,
                    help='The subject to process', required=False)
parser.add_argument('--run',  type=int, default=None,
                    help='which run', required=False)
args = parser.parse_args()
subject = f'sub-{args.subject:02}'
print('Processing subject:', subject)
# %%
run = None if args.run == None else f'{args.run:01}'
# Construct a raw object that will load the highpass-filtered data.
bids_path = BIDSPath(subject=subject[-2:], task=task, datatype='meg', processing='filt',
                     run=run,
                     root=fname.derivatives_dir)

raw = read_raw_bids(bids_path=bids_path, verbose=False)

# %%
# raw.crop(tmax=578) #subject 12 EOG1 didn't work from 578 s
# raw.crop(tmax=455) #subject 1
# %% fitting ica
n_components = 0.99
print('Fitting ICA')
ica = ICA(method='fastica', random_state=42, n_components=n_components)

#
raw.load_data()
# 1-40 hz raw for performing ICA
raw1 = raw.copy().filter(
    1, None, l_trans_bandwidth='auto',
    h_trans_bandwidth='auto', filter_length='auto', phase='zero',
    fir_window='hamming', fir_design='firwin', n_jobs=-1)

ica.fit(raw1,
        reject=dict(grad=4000e-13, mag=4e-12),
        decim=11)
# ica.fit(raw1, reject=dict(grad=4000e-13,), decim=11)#sub-12
print('Fit %d components (explaining at least %0.1f%% of the variance)'
      % (ica.n_components_, 100 * n_components))

ica.plot_sources(raw1, show_scrollbars=False)
# %%

bids_path = BIDSPath(subject=subject[-2:], task=task,
                     datatype='meg',
                     run=run,
                     root=fname.bids_dir
                     )

raw2 = read_raw_bids(bids_path=bids_path, verbose=False)
# %%
raw2.load_data()
raw2 = raw2.filter(
    0.1, 40, l_trans_bandwidth='auto',
    h_trans_bandwidth='auto', filter_length='auto', phase='zero',
    fir_window='hamming', fir_design='firwin', n_jobs=-1)
# %% find ecg events for plotting
# import matplotlib.pyplot as plt
# events, a, b, ecg = find_ecg_events(
#         raw2,
#         l_freq=8,
#         h_freq=16,
#         return_ecg=True,
#         reject_by_annotation=True,
#     )
# %%
event_id = 999
ecg_events, _, _, ecg = find_ecg_events(
    raw2,
    ch_name=None,
    event_id=event_id,
    l_freq=8,
    h_freq=16,
    return_ecg=True,
    reject_by_annotation=True,
)


# %%
# onsets = ecg_events[:, 0] / raw.info["sfreq"] - 0.25
# durations = [0.5] * len(ecg_events)
# descriptions = ["bad heart beats"] * len(ecg_events)
# blink_annot = mne.Annotations(
#     onsets, durations, descriptions, orig_time=raw.info["meas_date"]
# )
# raw1.set_annotations(blink_annot)
# raw1.plot(events=ecg_events)
# %%
picks = _picks_to_idx(raw.info, None, "all", exclude=())
# create epochs around ECG events and baseline (important)
ecg_epochs = Epochs(
    raw1,
    events=ecg_events,
    event_id=event_id,
    tmin=-.3,
    tmax=.3,
    proj=False,
    picks=picks,
    reject_by_annotation=True,
    preload=False,
)
ecg_evoked = ecg_epochs.average()
ecg_evoked.apply_baseline(baseline=(None, -0.2))

ecg_evoked.plot_joint()


# %%
# raw2.info["bads"] = []

# %%
# ica.plot_sources(raw, show_scrollbars=False)
# %% Find ICA components that correlate with heart beats.
ecg_epochs.decimate(5)
ecg_epochs.load_data()
ecg_epochs.apply_baseline((None, -0.2))
ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
# ecg_indices, ecg_scores = ica.find_bads_ecg(ecg_epochs, method="correlation", threshold="auto")
ecg_scores = np.abs(ecg_scores)
rank = np.argsort(ecg_scores)[::-1]
rank = [r for r in rank if ecg_scores[r] > 0.05]

print('    Found ECG indices', ecg_inds,)
print('    Found ECG components',  rank)
# print('    Found ECG components',  ecg_scores[rank[0]])
# %%

# %%
ica.exclude = rank[:n_ecg_components]
# %% Find ICA components that correlate with eye blinks
# Find onsets of heart beats and blinks. Create epochs around them
# ecg_epochs = create_ecg_epochs(raw2, tmin=-.3, tmax=.3, preload=False)
eog_epochs = create_eog_epochs(raw, tmin=-.5, tmax=.5, preload=False)
eog_epochs.decimate(5)
eog_epochs.load_data()
eog_epochs.apply_baseline((None, None))
eog_inds, eog_scores = ica.find_bads_eog(eog_epochs)
eog_scores = np.max(np.abs(eog_scores), axis=0)
# Look for components with a correlation > 0.1 to the EOG channels and that
# have not already been flagged as ECG components
rank = np.argsort(eog_scores)[::-1]
rank = [r for r in rank if eog_scores[r] > 0.1]
print('    Found EOG indices', eog_inds)
print('    Found EOG components', rank)
# %% find_eog_events
# eog_events = find_eog_events(
#     raw,
#     ch_name=None,
#     event_id=998,
#     l_freq=1,
#     h_freq=10,
#     reject_by_annotation=True,
#     thresh=None,
# )
# onsets = eog_events[:, 0] / raw.info["sfreq"] - 0.25
# durations = [0.5] * len(eog_events)
# descriptions = ["bad blink"] * len(eog_events)
# blink_annot = mne.Annotations(
#     onsets, durations, descriptions, orig_time=raw.info["meas_date"]
# )
# raw.set_annotations(blink_annot)
# raw.plot(events=eog_events)
# %%
ica.exclude += rank[:n_eog_components]

# %%
if args.run == None:
    ica.save(fname.ica(subject=subject), overwrite=True)
    title1 = 'ICA components'
    title2 = 'Component correlation'
else:

    ica.save(fname.ica1(subject=subject, run=args.run), overwrite=True)
    title1 = f'ICA components {args.run}'
    title2 = f'Component correlation {args.run}'
print('saved---------------------------------')
# Save plots of the ICA components to the report
# figs = ica.plot_components(show=False)

# %%Save plots of the ICA components to the report
with mne.open_report(fname.report(subject=subject)) as report:
    report.add_figure(
        ica.plot_components(show=False),
        title=title1,
        caption=['ICA components %d' % i for i in range(
            len(ica.plot_components(show=False)))],
        section='Sensor-level',
        replace=True
    )
    report.add_figure(
        [ica.plot_scores(ecg_scores, show=False),
          ica.plot_scores(eog_scores, show=False)],
        title=title2,
        caption=['Component correlation with ECG',
                  'Component correlation with EOG'],
        section='Sensor-level',
        replace=True
    )
    report.save(fname.report_html(subject=subject), overwrite=True,
                open_browser=False)
