#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tsss preprocessing

@author: jiaxin
"""
import os
# os.chdir()
import mne
import argparse
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids, get_anonymization_daysback
from config import fname, bad_channels, event_id, task

# %% Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--subject',  type=int, default=27,
                    help='The subject to process', required=False)
parser.add_argument('--runs',  type=int, default=1,
                    help='how many runs', required=False)
parser.add_argument('--run',  type=int, default=1,
                    help='which run', required=False)

args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

bads = bad_channels[f'sub-{subject:02}']
print('Bad channles:', bads)

# mne.set_log_level(verbose="warning")
tsss_log = fname.log(subject=subject, proc='tsss')
#
mne.set_log_file(tsss_log, overwrite=True)

if args.runs == 1:
    bids_path = BIDSPath(subject=f'{subject:02}', task=task,
                         datatype='meg',
                         root=fname.bids_dir
                         )
    write_bids_path = BIDSPath(subject=f'{subject:02}', task=task, datatype='meg',
                               processing='tsss',
                               root=fname.derivatives_dir)
    raw = read_raw_bids(bids_path=bids_path, verbose=False)

    dev_head_t_ref = raw.info['dev_head_t']
else:
    bids_path = BIDSPath(subject=f'{subject:02}', task=task,
                         datatype='meg',
                         run=f'{args.run:01}',
                         root=fname.bids_dir
                         )
    write_bids_path = BIDSPath(subject=f'{subject:02}', task=task, run=f'{args.run:01}',
                               datatype='meg', processing='tsss',
                               root=fname.derivatives_dir)

    bids_path1 = BIDSPath(subject=f'{subject:02}', task=task,
                          datatype='meg',
                          run=f'{1:01}',
                          root=fname.bids_dir
                          )

    # raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw = mne.io.read_raw_fif(bids_path, verbose=False)
    dev_head_t_ref = read_raw_bids(
        bids_path=bids_path1, verbose=False).info['dev_head_t']

# raw.plot(lowpass=40)#%%
# %%break annotationif
if subject == 1:
    onsets = [raw.first_time+477, raw.first_time+1048]
    durations = [18, 54]
    descriptions = ['BAD_ACQ_SKIP'] * len(onsets)
    break_annot = mne.Annotations(
        onsets, durations, descriptions, orig_time=raw.info["meas_date"]
    )
    raw.set_annotations(raw.annotations + break_annot)

#%%31.5.2024
raw.info["bads"] = []
auto_noisy_chs, auto_flat_chs, auto_scores = mne.preprocessing.find_bad_channels_maxwell(
         raw,
         cross_talk=fname.crosstalk,
         calibration=fname.fine_cal,
         return_scores=True,
         verbose=True,
     )
raw.info["bads"] = list(set(bads+ auto_noisy_chs + auto_flat_chs))
    
print(raw.info["bads"])


# %%
chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw, verbose=True)
chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes, verbose=True)
head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=True)


raw_tsss = mne.preprocessing.maxwell_filter(
    raw, cross_talk=fname.crosstalk, calibration=fname.fine_cal,
    origin='auto',
    st_duration=60,
    coord_frame='head',
    destination=dev_head_t_ref,

    head_pos=head_pos,
    verbose=True
)
raw_tsss = mne.chpi.filter_chpi(raw_tsss)
raw_tsss.drop_channels([f'CHPI{i:03d}' for i in range(1, 10)])


# %%%
event_id["BAD_ACQ_SKIP"] = 0
write_raw_bids(
    raw=raw_tsss,
    event_id=event_id,
    bids_path=write_bids_path,
    overwrite=True,
    format='FIF',
    allow_preload=True
)
