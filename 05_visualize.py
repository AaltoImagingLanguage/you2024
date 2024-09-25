#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 17:01:54 2023

@author: jiaxin
"""
# %% import modules
import argparse
import mne
from config import (fname, event_id, task)
import math
import numpy as np

import matplotlib.pyplot as plt
plt.ioff()
# %%
# cmps=plt.get_cmap('viridis').colors[0:256:85]
#
# cmps=list(plt.get_cmap('tab20b').colors[:4])
# cmps=list(plt.get_cmap('tab20b').colors[12:16])
# cmps = list(plt.get_cmap('tab20b').colors[16:])
# cmps.reverse()

# %%
# Be verbose
mne.set_log_level('INFO')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--subject',  type=int, default=1,
                    help='The subject to process', required=False)
args = parser.parse_args()
subject = f'sub-{args.subject:02}'
print('Processing subject:', subject)

# %% Read the epochs
print('Reading epochs...')
epochs = mne.read_epochs(fname.epo(subject=subject))
# %%
# epochs=epochs.crop(tmax=1)
# evks = [epochs[cat].average() for cat in event_id]
# # evks = [epochs.average()]
# # evokes = evks[0].data
# # a = np.sqrt(np.square(evokes).mean(0))
# # # plt.plot(source_estimate.times, a)
# # # %%
# fn = f'~/Xfit/{task}_{subject}-ave.fif'
# mne.write_evokeds(fn, evks, overwrite=True)
# %%
epochs = epochs.crop(tmax=1.1)
ave = {cat: epochs[cat].average().pick_types('grad') for cat in event_id}
fig, axes = plt.subplots(
    nrows=1, ncols=1, figsize=(25, 12), layout="constrained")
mne.viz.evoked.plot_evoked_topo([ave[i] for i in ave.keys()],
                                title=subject,
                                # layout_scale=1.5,
                                # color=cmps,
                                # scalings=dict(grad=3e13, 
                                              # mag=5e14
                                              # ),
                                ylim=dict(grad=[-40, 40]),
                                axes=axes,
                                show=False
                                )
# %% Save evoked plot to report
with mne.open_report(fname.report(subject=subject)) as report:
    report.add_figure(
        fig,
        title='Evoked topography',
        section='Sensor-level',
        replace=True
    )
    report.save(fname.report_html(subject=subject), overwrite=True)
# %%
# plt.savefig(f'/m/nbe/scratch/flexwordrec/scripts/figures/{subject}_topo_evokes1.pdf')
# %%
# for cat in event_id:
#     epoch_con=epochs[cat]
#     epoch_con.save(fname.epo_con(subject=subject,condition=cat), overwrite=True)
# %%
# event_id = {'RW': 1, 'RL1PW': 4, 'RL2PW': 5}
# ave = {cat: epochs[cat].average() for cat in event_id}
# %%
# ave = {cat: epochs[cat].average().pick_types('grad') for cat in event_id}
# ave = {cat: epochs[cat].average() for cat in event_id}
# evks = [epochs[cat].average() for cat in event_id]
# evks = [epochs.average()]
# evokes = evks[0].data
# a = np.sqrt(np.square(evokes).mean(0))
# # plt.plot(source_estimate.times, a)
# # %%
# fn = f'~/Xfit/{task}_sub-{subject:02d}-ave.fif'
# mne.write_evokeds(fn, evks, overwrite=True)
# %%

# %%
# ave = {cat: epochs[cat].average() for cat in event_id}
# for hemi in ['Left-']:
#     for loc in [
#         # 'parietal',
#             'temporal',
#         # 'frontal',
#         # 'occipital'
#     ]:
#         if loc == 'Vertex':
#             area = loc
#         else:
#             area = hemi+loc
#         selection = mne.read_vectorview_selection(area, info=epochs.info)
#         mne.viz.plot_compare_evokeds(
#               ave, picks=selection,
#               # show_sensors=True,
#                combine='gfp',
#               # axes='topo',
#               colors=cmps
#           )  
        # selection = mne.read_vectorview_selection(area, info=epochs.info)
        # mne.viz.plot_compare_evokeds(
        #     ave, picks=selection, show_sensors=True,
        #     # combine='gfp',
        #     axes='topo',
        #     colors=cmps
        # )
        # plt.savefig(f'figures/{subject}_{loc}.png')
# %%
# sensor_figs = []
# # sub-01
# # picks = ['MEG0233', 'MEG1612', 'MEG1523','MEG1623','MEG1642','MEG1933']

# # sub-02
# # picks = ['MEG1722', 'MEG1633','MEG0242']

# # sub-03
# # picks = ['MEG0243', 'MEG1613','MEG1632','MEG1642',]

# # sub-04
# # picks = ['MEG0222', 'MEG0233', 'MEG1722', 'MEG1732', 'MEG1642']
# # sub-05
# picks = ['MEG0222', 'MEG0232', 'MEG0242', 'MEG0122', 'MEG1523']

# sub-06
# picks = [ 'MEG0242', 'MEG1513',]


# #sub-11
# picks = [ 'MEG0242', 'MEG1513',]
# # picks = ['MEG1531', 'MEG1541', 'MEG1512', 'MEG0132', 'MEG0133', 'MEG0341', 'MEG0311', 'MEG0221', 'MEG0331', 'MEG0311',
# #          'MEG1512', 'MEG0413', 'MEG0321', 'MEG0121',]


# # picks = [
# #     'MEG0323', 'MEG1643', 'MEG1642', 'MEG0342', 'MEG1523',
# #     'MEG1633', 'MEG1613', 'MEG0243',
# #     'MEG1533',
# #     'MEG1632'
# # ]
# # picks = [
# #     'MEG1533',
# #     'MEG1632']
# for sensor in picks:
#     mne.viz.plot_compare_evokeds(
#         [ave[i] for i in ave.keys()],
#         picks=sensor,
#         colors=cmps
#         # show=True
#     )
#     # sensor_figs.extend(mne.viz.plot_compare_evokeds(
#     #     [ave[i] for i in ave.keys()],
#     #     picks=sensor,
#     #     show=True
#     # ))
#     plt.savefig(f'./figures/{subject}_{sensor}.png')
# # %%
# with mne.open_report(fname.report(exp_type=args.exp_type, subject=subject)) as report:
#     report.add_figure(
#         sensor_figs,
#         title='Evokeds in select channels',
#         section='Sensor-level',
#         replace=True
#     )
#     report.save(fname.report_html(exp_type=args.exp_type, subject=subject), overwrite=True,
#                 open_browser=False)

# # %%
# f = '/m/nbe/archive/flexwordrec/MEG/flexw_sub-02/231013/sub002_flexw_ave.fif'
# ave = mne.read_evokeds(f)
# mne.viz.evoked.plot_evoked_topo(ave,
#                                 scalings=dict(grad=3e13, mag=5e14),
#                                 # ylim=dict(grad=[-30, 30]),
#                                 )
# %%
