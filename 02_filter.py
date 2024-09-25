#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter tsss-preprocessed data
"""
# %%
import mne
import argparse
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids
from config import fname, bandpass_fmin, bandpass_fmax, task, bad_channels, phase, event_id


# %%
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--subject',  type=int, default=27,
                    help='The subject to process')
parser.add_argument('--run',  type=int, default=None,
                    help='which run', required=False)
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

mne.set_log_level(verbose="info")
log = fname.log(subject=subject, proc='filter')
#
mne.set_log_file(log, overwrite=True)
# Keep track of PSD plots before and after filtering
# Load the tSSS transformed data.
run = None if args.run == None else f'{args.run:01}'
bids_path = BIDSPath(subject=f'{subject:02}', task=task, datatype='meg',
                     run=run,
                     processing='tsss',
                     root=fname.derivatives_dir)
raw = read_raw_bids(bids_path=bids_path, verbose=False)

# raw.plot(lowpass=40)

event_id['BAD_ACQ_SKIP'] = 0
# %%
raw.load_data()
raw_filt = raw.copy().filter(
    bandpass_fmin, bandpass_fmax, l_trans_bandwidth='auto',
    # method= 'iir',
    # phase=phase,
    h_trans_bandwidth='auto', filter_length='auto', phase='zero',
    fir_window='hamming', fir_design='firwin',
    n_jobs=1
    )

# Highpass the EOG channels to > 1Hz, regardless of the bandpass-filter
# applied to the other channels
picks_eog = mne.pick_types(raw_filt.info, meg=False, eog=True)
raw_filt.filter(
    1., None, picks=picks_eog, l_trans_bandwidth='auto',
    filter_length='auto', phase='zero', fir_window='hann',
    fir_design='firwin',
    n_jobs=1
)
raw_filt.notch_filter(freqs=50, picks=picks_eog)
# raw_filt.plot()

# %%
bids_path = BIDSPath(subject=f'{subject:02}', task=task, datatype='meg',
                     run=run,
                     processing='filt',
                     root=fname.derivatives_dir)
write_raw_bids(
    raw=raw_filt,
    bids_path=bids_path,
    allow_preload=True,
    format='FIF',
    event_id=event_id,
    overwrite=True

)
fig_before = raw.plot_psd(show=False)
fig_after = raw_filt.plot_psd(show=False)

# %% Append PDF plots to report
with mne.open_report(fname.report(subject=f'sub-{subject:02}')) as report:
    report.add_figure(
        fig_before,
        title=f'PSD before filtering ({args.run})',
        section='Sensor-level',
        replace=True
    )
    report.add_figure(
        fig_after,
        title=f'PSD after filtering ({args.run})',
        section='Sensor-level',
        replace=True
    )
    report.save(fname.report_html(subject=f'sub-{subject:02}'), overwrite=True,
                open_browser=False)
