
"""
Cut signal into epochs.
"""
# %% import modules
import argparse
import mne
import pandas as pd
import numpy as np
from mne.preprocessing import read_ica
from mne_bids import BIDSPath, read_raw_bids
from config import (fname,  event_id,
                    epoch_tmin, epoch_tmax, baseline, reject, task, bad_channels, phase)

# %%
# Be verbose
mne.set_log_level('INFO')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--subject',  type=int, default=16,
                    help='The subject to process', required=False)
parser.add_argument('--runs',  type=int, default=1,
                    help='how many runs', required=False)
args = parser.parse_args()
subject = f'sub-{args.subject:02}'
print('Processing subject:', subject)
print(fname.report_html(subject=subject))
# %%combine epochs
epochs_list = []
for run in range(args.runs):
    run = f'{run+1:01}' if args.runs > 1 else None
# Construct a raw object that will load the highpass-filtered data.
    bids_path = BIDSPath(subject=subject[-2:], task=task, datatype='meg', processing='filt',
                         run=run,
                         root=fname.derivatives_dir)
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    # raw.load_data()
    # bads = bad_channels[subject]
    # raw.info["bads"] = []
    # raw.info["bads"] = bads
    # raw.plot()
    # Load the ICA object
    print('  Using ICA')
    if run == None:
        ica = read_ica(fname.ica(subject=subject))
        title = 'Evoked without ICA'
    else:
        ica = read_ica(fname.ica1(subject=subject, run=run))
        title = f'Evoked without ICA (run-{run})'
    #
    # ica.apply(raw)
    # raw.plot()
    #
    events, _ = mne.events_from_annotations(raw, event_id)
    if subject == 'sub-03':
        events = events[1:, :]
    if subject == 'sub-01':
        index_delete = []
        for j in range(events.shape[0]-1):
            event_n0 = events[j, 0]
            event_n1 = events[j+1, 0]
            if (event_n1-event_n0) > 3300 and (event_n1-event_n0) < 7000:
                # if (event_n1-event_n0)>3300:
                index_delete.append(j)
        print('index_delete length', len(index_delete))
        events = np.delete(events, index_delete, axis=0)
    print(events.shape)
    # =============================================================================
    # # Read events from the stim channel
    # mask = 4096 + 256  # mask for excluding high order bits
    # events1 = mne.find_events(raw, stim_channel='STI101', consecutive='increasing',
    #                           mask=mask, mask_type='not_and', min_duration=0.003)
    # # todo: compare events and events1!
    # =============================================================================
    #
    projector_delay = 0.010  # For the new (2019-09 onwards) Aalto setup
    events = mne.event.shift_time_events(events, list(event_id.values()),
                                         projector_delay, raw.info['sfreq'])


    #  Make epochs.
    # Because the original 1000Hz sampling rate is a bit excessive
    # for what we're going for, we only read every 2th sample. This gives us a
    # sampling rate of ~500Hz.
    epochs = mne.Epochs(raw, events, event_id, epoch_tmin, epoch_tmax,
                        reject_by_annotation=False,
                        baseline=baseline, decim=2,
                        # reject=dict(grad=4000e-13,
                        # mag=4e-12
                        # ),
                        # metadata=metadata,
                        preload=True)

    # Apply ICA to the epochs, dropping components that correlate with ECG and EOG
    ica.apply(epochs)
    epochs.apply_baseline(baseline)
    # Drop epochs that have too large signals (most likely due to the subject
    # moving or muscle artifacts)

    # plot epochs before ica
    with mne.open_report(fname.report(subject=subject)) as report:
        report.add_figure(
            [epochs.average().plot(show=False)],
            title=title,
            section='Sensor-level',
            replace=True
        )
        report.save(fname.report_html(
                    subject=subject), overwrite=True)

    epochs_list.append(epochs)

# %% concatenate epochs if there are more than 1 run
if epochs_list == 1:
    epochs = epochs_list[0]
else:
    epochs = mne.concatenate_epochs(epochs_list)
# %%write metadata
stimuli = pd.read_csv(f'{fname.data_dir}/stimuli_metadata.csv')
if subject == 'sub-01':
    stimuli = stimuli.drop(
        index=[200, 201, 202, 203, 204, 205])  # # incude 5 regular trials and 1 catch trial
        # index=[212, 213, 214, 215, 216, 217])  
metadata = stimuli[stimuli['target'] == '0']  # exclude catch trials

#%%
if subject == 'sub-17':
    metadata = metadata.drop([309])#309 index of the df, not the row index(279)
if subject == 'sub-03':
    metadata = metadata.drop([0])
epochs.metadata = metadata
print('  Dropped %0.1f%% of epochs' % (epochs.drop_log_stats(),))


epochs.drop_bad(reject)
print('  Writing to disk')
epochs.save(fname.epo(subject=subject), overwrite=True)

# %% Save evoked plot to the report


# %%
for cat in event_id:
    epoch_con = epochs[cat]
    epoch_con.save(fname.epo_con(subject=subject,
                   condition=cat), overwrite=True)


# Save evoked plot to report
with mne.open_report(fname.report(subject=subject)) as report:
    report.add_figure(
        [epochs.average().plot(show=False)],
        title='Evoked with ICA',
        section='Sensor-level',
        replace=True
    )
    report.save(fname.report_html(subject=subject), overwrite=True)

#%% check 
# e1=metadata['# Replaced letters']+1
# e1=np.array(e1[:-1])#"1" numbers are different
# e2=events[:,2]
# e=e2-e1[:-1]
# index=e.tolist().index(-1)#"-1" is the fisrt element that are different from 0
# np.delete(e1, index)
