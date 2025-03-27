#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config parameters
"""
# %%
import os
from filename_templates import FileNames
import getpass
from socket import getfqdn
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from scipy import stats as stats
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import numpy as np
# %%
user = getpass.getuser()  # Username of the user running the scripts
host = getfqdn()  # Hostname of the machine running the scripts

if user == "jiaxin":
    study_path = "/scratch/flexwordrec"
    derivatives_dir = "/scratch/flexwordrec/bids/derivatives"
    analysis_dir = "/m/nbe/scratch/flexwordrec"
    FREESURFER_HOME = '/usr/local/freesurfer/7.4.0'
elif user == "vanvlm1":
    study_path = "/m/nbe/scratch/flexwordrec"
    derivatives_dir = "/m/nbe/scratch/flexwordrec/marijn/bids/derivatives"
    analysis_dir = "/m/nbe/scratch/flexwordrec/marijn"
    FREESURFER_HOME = '/usr/local/freesurfer/7.4.1'
    # derivatives_dir = "/scratch/flexwordrec/bids/derivatives"
else:
    study_path = "/m/nbe/scratch/flexwordrec"
    # derivatives_dir = "/m/nbe/scratch/flexwordrec/bids/derivatives"
    FREESURFER_HOME = '/work/modules/Ubuntu/20.04/amd64/t314/freesurfer/dev-20201103-bd997c7'
    derivatives_dir = "/m/nbe/scratch/flexwordrec/bids/derivatives"
    analysis_dir = "/m/nbe/scratch/flexwordrec"

subjects = [

    'sub-01',
    'sub-02',
    'sub-03',
    'sub-04',
    'sub-05',
    'sub-06',
    'sub-07',
    # 'sub-08',#tattoo around face area
    'sub-09',
    # 'sub-10',
    'sub-11',
    'sub-12',
    'sub-13',
    # 'sub-14',
    'sub-15',
    'sub-16',
    'sub-17',
    'sub-18',
    'sub-19',
    'sub-20',
    'sub-21',
    'sub-22',
    # 'sub-23',#double-vision problem
    'sub-24',
    'sub-25',
    'sub-26',
    'sub-27',

]
parc = 'aparc.a2009s_custom_gyrus_sulcus_800mm2'
grow_rois_seeds = {'tc'}
# %%% relevant parameters for the analysis.
task = 'flexwordrec'
# Band-pass filter limits. Since we are performing ICA on the continuous data,
# it is important that the lower bound is at least 1Hz.
bandpass_fmin = 0.1  # Hz
bandpass_fmax = 40  # Hz

# Maximum number of ICA components to reject
n_ecg_components = 1  # ICA components that correlate with heart beats
n_eog_components = 2  # ICA components that correlate with eye blinks

# Time window (relative to stimulus onset) to use for extracting epochs
epoch_tmin, epoch_tmax = -0.2, 1.1

# Time window to use for computing the baseline of the epoch
baseline = (-0.2, 0)

# Thresholds to use for rejecting epochs that have a too large signal amplitude
reject = dict(grad=3E-10, mag=4E-12)

# marked bad channels during measurments
bad_channels = {
    # 'pilot1': ['MEG0313', 'MEG0723', 'MEG2542'],
    # 'pilot2': ['MEG2142', 'MEG0723', 'MEG2322', 'MEG0313'],
    # 'pilot3': ['MEG2142', 'MEG0723', 'MEG0313', 'MEG2542'],
    'sub-01': ['MEG2142', 'MEG0723', 'MEG2322', 'MEG0313'],
    'sub-02': ['MEG2142', 'MEG0723', 'MEG2322', 'MEG0313', 'MEG1422'],
    'sub-03': ['MEG2142', 'MEG0723', 'MEG0313'],
    'sub-04': ['MEG2142', 'MEG0723', 'MEG0313', 'MEG2542', 'MEG0532'],
    'sub-05': ['MEG0723', 'MEG2142', 'MEG0313', 'MEG0532'],
    'sub-06': ['MEG0723', 'MEG0742', 'MEG0313'],
    'sub-07': ['MEG0723', 'MEG0313', 'MEG0812', 'MEG0532', 'MEG2322', 'MEG2142', 'MEG2442', 'MEG1322'],
    # 'sub-08': ['MEG0723', 'MEG0313','MEG0532', 'MEG2322', 'MEG2542'],
    'sub-09': ['MEG0723', 'MEG0313', 'MEG2542', 'MEG0532',],
    'sub-10': ['MEG0723', 'MEG0313','MEG0812','MEG2322','MEG2132','MEG2542'],
    # 'sub-11': ['MEG0723', 'MEG0313','MEG0812','MEG2322','MEG2542'],
    'sub-12': ['MEG0723', 'MEG0313', 'MEG0812', 'MEG2322', 'MEG2542', 'MEG2442'],
    'sub-13': ['MEG0723', 'MEG0313', 'MEG0812', 'MEG2322', 'MEG2542'],
    # 'sub-14': ['MEG0723', 'MEG0313','MEG0812','MEG2322','MEG2542','MEG2442','MEG0532'],
    'sub-15': ['MEG0723', 'MEG0313', 'MEG0812', 'MEG2322', 'MEG0532'],
    'sub-16': ['MEG0723', 'MEG0313', 'MEG2322', 'MEG0532', 'MEG2542'],
    'sub-17': ['MEG0723', 'MEG0313', 'MEG2322', 'MEG0532', 'MEG2542'],
    'sub-18': ['MEG0723', 'MEG2322', 'MEG2542', 'MEG0313', 'MEG1212',],
    'sub-19': ['MEG0723', 'MEG0313', 'MEG0812', 'MEG2442', 'MEG2322', 'MEG0742', 'MEG0532'],
    'sub-20': ['MEG0723', 'MEG1212', 'MEG2322', 'MEG0313'],
    'sub-21': ['MEG0723', 'MEG1212', 'MEG2322', 'MEG0313'],
    'sub-22': ['MEG0723', 'MEG0812', 'MEG2322', 'MEG0313'],
    'sub-23': ['MEG0723', 'MEG2542', 'MEG1212', 'MEG0313'],
    'sub-24': ['MEG0723', 'MEG2542', 'MEG1212', 'MEG0313'],
    'sub-25': ['MEG0723', 'MEG0313', 'MEG0812', 'MEG1933','MEG2333'],
    'sub-26': ['MEG0723', 'MEG0313', 'MEG0812', 'MEG2542'],
    'sub-27': ['MEG0723', 'MEG0313', 'MEG0812', 'MEG1322'],



}

eog_chs = ['EOG001', 'EOG002']

# The event codes used in the experimen
event_id = {"RW": 1, "RL1PW": 2, "RL2PW": 3, "RL3PW": 4}


# Time window (relative to stimulus onset) to use for computing the CSD
csd_tmin, csd_tmax = 0.35, 0.4
# csd_tmin, csd_tmax = 0, 0.7

# Spacing of sources to use
spacing = 'ico4'

# Maximum distance between sources and a sensor (in meters)
max_sensor_dist = 0.07

# Minimum distance between sources and the skull (in mm)
min_skull_dist = 0

# Regularization parameter to use when computing the DICS beamformer
reg = 0.05

# Frequency bands to perform powermapping for
freq_bands = [
    (3, 7),     # theta
    (7, 13),    # alpha
    (13, 17),   # low beta
    (17, 25),   # high beta 1
    (25, 31),   # high beta 2
    (31, 40),   # low gamma
    (40, 90),   # high gamma
]

# Frequency band to use when computing connectivity (low gamma)
con_fmin = 31
con_fmax = 40

# Minimum distance between sources to compute connectivity for (in meters)
min_pair_dist = 0.04

n_jobs = -1
# Regularization parameter to use when computing the DICS beamformer
reg = 0.05
phase = "zero"

# %%
window_length=75 # for smooth
cmap = mpl.cm.magma
cmaps4 = [cmap(i) for i in np.linspace(0, 0.8, 4)]
roi_colors =list(plt.get_cmap('tab10').colors[:3])#for rois color
rois_names=["pC","ST",'vOT',]
rois_id=[82,65,40]

# %%% Templates for filenames
fname = FileNames()

# Some directories
fname.add('MEG_path', '/m/nbe/archive/flexwordrec/MEG')

fname.add('study_path', study_path)
fname.add('FREESURFER_HOME', FREESURFER_HOME)
fname.add('bids_dir', '{study_path}/bids')
fname.add("derivatives_dir", derivatives_dir, mkdir=True)
fname.add("analysis_dir", analysis_dir, mkdir=True)

fname.add('mri_subjects_dir', '{analysis_dir}/mri_subjects/', mkdir=True)
fname.add('subjects_dir', '{analysis_dir}/subjects/', mkdir=True)
fname.add('figures_dir', '{analysis_dir}/figures/{subject}/', mkdir=True)
fname.add('meg_dir', '{analysis_dir}/MEG', mkdir=True)
fname.add('anatomy', '{mri_subjects_dir}/{subject}', mkdir=True)
fname.add('bem_dir', '{anatomy}/bem', mkdir=True)
fname.add('sp', spacing)  # Add this so we can use it in the filenames below
fname.add('src', '{anatomy}/fsaverage_to_{subject}-{sp}-src.fif')
fname.add('fsaverage_src',
          '{mri_subjects_dir}/fsaverage/fsaverage-{sp}-src.fif')


fname.add('subject_dir', '{bids_dir}/sub-{subject:02d}')
fname.add(
    'raw', '{subject_dir}/meg/sub-{subject:02d}_task-flexwordrec_run_{run}_meg.fif')

fname.add('cal_path', '/m/nbe/scratch/flexwordrec/calibration_files')
fname.add('fine_cal', '{cal_path}/sss_cal_Aalto_TRIUXneo_3158.dat')
fname.add('crosstalk', '{cal_path}/ct_sparse_Aalto_TRIUXneo_3158.fif')

fname.add(
    'log', '{analysis_dir}/logs/sub-{subject}_{proc}_log.txt', mkdir=True)


fname.add('ica', '{subjects_dir}/{subject}_ica.fif')
# ica for per run
fname.add('ica1', '{subjects_dir}/{subject}_run-{run}_ica.fif')
fname.add('epo', '{subjects_dir}/{subject}-epo.fif')
fname.add('epo_con', '{subjects_dir}/{subject}-{condition}-epo.fif')
fname.add('csd', '{subjects_dir}/{subject}-{condition}-csd.h5')
fname.add('power', '{subjects_dir}/{subject}-{condition}-dics-power')
fname.add('trans', '{subjects_dir}/{subject}-trans.fif')
fname.add('fwd', '{subjects_dir}/fsaverage_to_{subject}-meg-{sp}-fwd.fif')
fname.add('fwd_r', '{subjects_dir}/{subject}-{sp}-fwd.fif')
fname.add('inv', '{subjects_dir}/{subject}-{sp}-inv.fif')
fname.add('inv1', '{subjects_dir}/{subject}-{sp}1-inv.fif')
# fname.add('src', '{subjects_dir}/{subject}-{sp}-src.fif')

fname.add('src', '/m/nbe/scratch/flexwordrec/mri_subjects/{subject}-{sp}-src.fif')
fname.add("stc", "{subjects_dir}/{subject}_{category}_stc")
fname.add("stc_epos", "{subjects_dir}/{subject}_stc_epos")
fname.add("stc_morph", "{subjects_dir}/{subject}_{category}_morph_stc")
fname.add("stc_cpt", "{subjects_dir}/{subject}_{category}-RW_cpt_stc")
fname.add("ga_stc", "{subjects_dir}/grand_average_{category}_stc")
fname.add("ga_stc1", "{subjects_dir}/grand_average_{category}1_stc")
fname.add("rsa", "{subjects_dir}/{subject}/rsa/{subject}_{category}_rsa_stc")
fname.add(
    "rsa_morph", "{subjects_dir}/{subject}/rsa/{subject}_{category}_rsa_morph_stc")
fname.add(
    "ga_rsa", "{subjects_dir}/fsaverage/rsa/fsaverage_{category}_rsa_stc")

fname.add('pairs', '{meg_dir}/pairs.npy')
# fname.add('epo', '{study_path}/subjects/pilot_sub-{subject:02d}-epo.fif')

# Filenames for MNE reports
fname.add('reports_dir', '{analysis_dir}/reports/')
fname.add('report', '{reports_dir}/{subject}-report.h5')
fname.add('report_html',
          '{reports_dir}/{subject}-report.html')

# %%Time-lagged MDPC

# for inverse operator
f_down_sampling = 20

snr = 3.
lambda2 = 1.0 / snr ** 2

snr_epoch = 3.
lambda2_epoch = 1.0 / snr_epoch ** 2
n_permutations = 5000
tail = 0
pvalue = 0.05
# len(bad_channels)==len(subjects)
t_threshold = -stats.distributions.t.ppf(pvalue / 2.0, len(bad_channels) - 1)

colors1 = [
    "dimgrey",
    "darkred",
    "red",
    "orange",
    "gold",
    "yellow",
    "white",
]


colors2 = ["blue", "green", "white", "yellow", "red"]
colors3 = ["darkgrey", "grey", "dimgrey",
           "black", "dimgrey", "grey", "darkgrey"]

background_color = "white"
font_color = "black"


cmap_name = "my_list"
n_bin = 100
cm1 = LinearSegmentedColormap.from_list(cmap_name, colors1, N=n_bin)
cm2 = LinearSegmentedColormap.from_list(cmap_name, colors2, N=n_bin)
cm3 = LinearSegmentedColormap.from_list(cmap_name, colors3, N=n_bin)
#
fname.add('label_path',
          '{mri_subjects_dir}/fsaverage/label')
fname.add('mdpc_dir', '{derivatives}/mdpc/')
