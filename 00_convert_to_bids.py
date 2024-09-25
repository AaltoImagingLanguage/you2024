#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert raw MEG data to BIDS format
"""
# %% import modules
import json
import os.path as op
from pprint import pprint
import shutil
import os
import mne
import argparse
import numpy as np
from config import fname, event_id, task, bad_channels
from mne_bids import (write_raw_bids, get_anonymization_daysback,
                      BIDSPath, print_dir_tree)
from mne_bids.stats import count_events
# %% Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--subjects',  type=int, default=[1],
                    help='The list of subjects to process', required=False)
args = parser.parse_args()

# %% search the target raw files
# all folders under MEG folder
subj_list = os.listdir(fname.MEG_path)

a = subj_list.copy()
for i in range(len(a)):
    if os.path.isfile(os.path.join(fname.MEG_path, a[i])):
        subj_list.remove(a[i])
del a

# select the subjects
subj_list.remove('pilots')  # [flexw_sub-xx,...]
subjects = [subj for subj in subj_list if int(subj[-2:]) in args.subjects]

# %% Prepare bids paths
raw_list = list()
bids_list = list()
events_list = list()
event_id["BAD_ACQ_SKIP"] = 0
for i, sub in enumerate(subjects):
    file_path = os.path.join(fname.MEG_path, sub)
    runs = []
    for (root, dirs, file) in os.walk(file_path):
        file = [
            f for f in file if 'raw.fif' in f and "rest" not in f and 'empty' not in f]
        for run, f in enumerate(file):

            print('os.path.join(root, f):', os.path.join(root, f))
            # only a whole run
            if len(file) == 1:
                bids_path = BIDSPath(subject=f'{args.subjects[i]:02}', task=task,
                                     datatype='meg',
                                     root=fname.bids_dir)
            # multiple runs
            elif len(file) > 1:
                bids_path = BIDSPath(subject=f'{args.subjects[i]:02}', task=task,
                                     run=f'{run+1:01}', datatype='meg',
                                     root=fname.bids_dir)
            else:
                raise Exception("Sorry, no raw files")

            bids_list.append(bids_path)
            raw = mne.io.read_raw_fif(os.path.join(root, f))
            # raw.plot(lowpass=40)
            bads = bad_channels[f'sub-{args.subjects[i]:02}']
            raw.info["bads"] = []
            raw.info["bads"] = bads
            raw_list.append(raw)
            events = mne.find_events(raw,min_duration=0.005)
            events = events[(events[:, -1] < 5), :]
            # events=events[1:,:] #sub-03
            print('events', events.shape)

            # index_delete=[]
            # for j in range(events.shape[0]-1):
            #     event_n0=events[j,0]
            #     event_n1=events[j+1,0]
            #     if (event_n1-event_n0)>3300 and (event_n1-event_n0)<7000:
            #     # if (event_n1-event_n0)>3300:
            #         index_delete.append(j)
            # print('index_delete length',len(index_delete))
            # events=np.delete(events,index_delete, axis=0)
            # print('events',events.shape)

            events_list.append(events)


# %% write raw bids
daysback_min, daysback_max = get_anonymization_daysback(raw_list)
for raw, bids_path, events in zip(raw_list, bids_list, events_list):
    # By using the same anonymization `daysback` number we can
    # preserve the longitudinal structure of multiple sessions for a
    # single subject and the relation between subjects. Be sure to
    # change or delete this number before putting code online, you
    # wouldn't want to inadvertently de-anonymize your data.
    #
    # Note that we do not need to pass any events, as the dataset is already
    # equipped with annotations, which will be converted to BIDS events
    # automatically.
    print(raw, bids_path, events)
    write_raw_bids(
        raw=raw,
        bids_path=bids_path,
        events=events,
        event_id=event_id,
        anonymize=dict(daysback=daysback_min + 2117),
        overwrite=True)
    print('Done!')


