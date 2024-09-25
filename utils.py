import numpy as np
import mne
from config import fname, event_id
import matplotlib.pyplot as plt
# from config import roinames_time
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from functools import partial
import types
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mne.viz import Brain
from mne.viz.topo import _iter_topography
import numpy as np
from scipy.spatial import distance
import matplotlib as mpl






def create_grow_ROIs(stcs, roinames_time, extents=20):

    func_rois = []
    for roiname in roinames_time:

        if roiname == 'STC':
            labels_name = ['superiortemporal-lh',
                           'bankssts-lh', 'transversetemporal-lh']
        elif roiname == 'OTC':
            labels_name = ['fusiform-lh']
        elif roiname == 'precentral':
            labels_name = ['precentral-lh']
        elif roiname == 'IFG':
            labels_name = ['parsorbitalis-lh', 'parstriangularis-lh', 'parsopercularis-lh',
                           'lateralorbitofrontal-lh', 'frontalpole-lh']

        tmin, tmax = roinames_time[roiname][:-1]
        stc = stcs[roinames_time[roiname][-1]]-stcs[0]

        peak_vertex, peak_time = stc.get_peak(
            hemi='lh', tmin=tmin, tmax=tmax, mode='pos')  # vertex ID

        rough_labels = mne.grow_labels(
            'fsaverage', [peak_vertex], extents, hemis=0, names=roiname)

        brain=stc.plot(initial_time=peak_time,hemi='lh',)

        # #%%
        brain.add_foci(peak_vertex, coords_as_verts=True, hemi="lh", color="k")
        # # We add a title as well, stating the amplitude at this time and location
        # brain.add_text(0.1, 0.9, "Peak coordinate", "title", font_size=14)

        # # rough_labels=mne.label.select_sources(SUBJECT,'lh',peak_vertex, extents,subjects_dir=fname.mri_subjects_dir, name=roiname)

        # brain.add_label(rough_labels[0])

        func_rois.append(rough_labels[0])

    return func_rois


def create_grow_ROIs1(stcs, roinames_time=['STC1','STC2','precentral','OT'], extents=15,plot=False):
    labels = mne.read_labels_from_annot('fsaverage', 'aparc.a2009s', 'lh',
                                        )
    func_rois = []
    stc = stcs[-1]-stcs[0]
    for roiname in roinames_time:
        print(roiname)
        if roiname == 'STC1':
            labels_name = ['G_temp_sup-G_T_transv-lh',
                           'S_temporal_transverse-lh',]
            extents=20
        elif roiname == 'STC2':
            labels_name = ['S_temporal_sup-lh',]
            extents=15
        elif roiname == 'precentral':
            labels_name = ['G_precentral-lh','S_central-lh','S_precentral-inf-part-lh']
            extents=15
        elif roiname == 'OT':
            labels_name = ['S_oc-temp_lat-lh']
            stc = stcs[-2]-stcs[0]
            extents=15
        print(labels_name)
        for j in range(0, len(labels_name)):
            print(j)
            l = [label for label in labels if label.name ==
                 labels_name[j]][0]
            if j == 0:
                label = l
            else:
                label += l

        # stc = np.mean(stcs)

        stc_label = stc.in_label(label)
        peak_vertex, peak_time = stc_label.get_peak(
            hemi='lh',
            tmin=0.3, tmax=1.1,
            mode='pos')  # vertex ID

        rough_labels = mne.grow_labels(
            'fsaverage', [peak_vertex], extents, hemis=0, names=roiname)

        if plot:

            brain = stc.plot(initial_time=peak_time, hemi='lh',)
            
            #%%
            brain.add_foci(peak_vertex,
                            coords_as_verts=True,
                           hemi="lh", color="k")
            # # We add a title as well, stating the amplitude at this time and location
            # brain.add_text(0.1, 0.9, "Peak coordinate", "title", font_size=14)
    
            rough_labels = mne.grow_labels(
                'fsaverage', [peak_vertex], extents, hemis=0, names=roiname)
    
            brain.add_label(rough_labels[0], borders=True)
            print('.')

        func_rois.append(rough_labels[0])

    return func_rois

def create_grow_ROIs2(stcs, roinames_time=['OTC','STC','precentral','OC'], 
                      parc='aparc.a2009s_custom_gyrus_sulcus_1100mm2',
                      extents=30, plot=False):
    labels = mne.read_labels_from_annot('fsaverage', parc, 'both',
                                        )
    func_rois = []
    for roiname in roinames_time:

        if roiname == 'STC':
            stc = stcs[-1]-stcs[0]
            labels_name = [ 
                
                # 'G_temp_sup-Lateral+S_temporal_sup_sub1-lh',
                # 'G_temp_sup-Lateral+S_temporal_sup_sub2-lh',
                #   'G_temp_sup-Lateral+S_temporal_sup_sub3-lh',
                'G_pariet_inf-Supramar+G_temp_sup-Plan_tempo+G_temp_sup-G_T_transv+S_temporal_transverse+Lat_Fis-post_sub1-lh',
                # 'G_Ins_lg_and_S_cent_ins+G_temp_sup-Plan_polar+S_circular_insula_inf+G_insular_short+S_circular_insula_ant_sub2-lh',
                ]
        elif roiname == 'OTC':
            labels_name = ['G_oc-temp_lat-fusifor+S_oc-temp_lat-lh',
                           # 'G_and_S_occipital_inf+S_occipital_ant+S_collat_transv_post-lh'
                           ]
            
            stc = stcs[2]-stcs[0]
        elif roiname == 'OC':
            labels_name = [
                
                            'G_and_S_occipital_inf+S_occipital_ant+S_collat_transv_post-lh'
                           ]
            # extents=15
            stc = stcs[2]-stcs[0]
        elif roiname == 'precentral':
            # extents=15
            stc = stcs[-1]-stcs[0]
            labels_name = [ 'G_precentral+S_central+S_precentral-inf-part+S_precentral-sup-part_sub2-lh',
                           'G_precentral+S_central+S_precentral-inf-part+S_precentral-sup-part_sub1-lh',]
        # elif roiname == 'IFG':
        #     labels_name = ['parsorbitalis-lh', 'parstriangularis-lh', 'parsopercularis-lh',
        #                    'lateralorbitofrontal-lh', 'frontalpole-lh']

        for j in np.arange(0, len(labels_name)):
            l = [label for label in labels if label.name ==
                 labels_name[j]][0]
            if j == 0:
                label = l
            else:
                label += l
        # tmin, tmax = roinames_time[roiname][:-1]
        # stc = np.mean(stcs)

        stc_label = stc.in_label(label)
        peak_vertex, peak_time = stc_label.get_peak(
            hemi='lh',
            # vert_as_index=False,
            tmin=0.3, tmax=1.1,
            mode='pos')  # vertex ID
        # peak_vertex, _, peak_time=stc_label.center_of_mass(SUBJECT)
        
        rough_labels = mne.grow_labels(
            'fsaverage', [peak_vertex], extents, hemis=0, names=roiname)
        if plot:

            brain = stc.plot(initial_time=peak_time, hemi='lh',)
            
            #%%
            brain.add_foci(peak_vertex,
                            coords_as_verts=True,
                           hemi="lh", color="k")
            # # We add a title as well, stating the amplitude at this time and location
            # brain.add_text(0.1, 0.9, "Peak coordinate", "title", font_size=14) 
    
            brain.add_label(rough_labels[0], borders=True)
            print('.')

        func_rois.append(rough_labels[0])

    return func_rois

def stc_baseline_correction(X, stc, tmin, tmax):
    time_dim = len(stc.times)
    # baseline_timepoints = X.times[np.where(X.times<0)]
    # baseline_timepoints = X.times[np.where(X.times==tmin):np.where(X.times==tmax)]
    # Convert tmin/tmax to sample indices
    tmin, tmax = np.searchsorted(stc.times, [tmin, tmax])

    baseline_timepoints = stc.times[tmin:tmax]

    baseline_mean = X[:, tmin:tmax].mean(1)

    baseline_mean_mat = np.repeat(baseline_mean.reshape([len(baseline_mean), 1]),
                                  time_dim, axis=1)
    corrected_stc = X - baseline_mean_mat
    return corrected_stc


def mask_function(X, cut_off=None):
    if cut_off is not None:
        r, c = X.shape
        for i in np.arange(0, r):
            for j in np.arange(0, c):
                if X[i, j] < cut_off:
                    X[i, j] = cut_off
    return X

def select_rois(rois_id, parc='aparc.a2009s_custom_gyrus_sulcus_1100mm2',combines=[[0,1]]):
    annotation = mne.read_labels_from_annot(
        'fsaverage', parc=parc,)
    rois = [label for label in annotation if 'Unknown' not in label.name]
    labels=[]
    if combines:
        for ids in combines:
            for j in np.arange(0, len(ids)):
                if j == 0:
                    label = rois[rois_id[ids[j]]]
                else:
                    label += rois[rois_id[ids[j]]]
            labels.append(label)
    com_ids=[rois_id[i] for i in sum(combines, [])]
    left_ids=[i for i in rois_id if i not in com_ids]
    labels.extend([rois[i] for i in left_ids])
    return labels
    

def compare_p(pval):
    """
    By: youj2

    Parameters
    ----------
    pval : TYPE
        DESCRIPTION.

    Returns
    -------
    p : TYPE
        DESCRIPTION.

    """
    if pval < 0.001:
        p = 0.001
        stars="***"
    # elif pval < 0.005:
    #     p = 0.005
    elif pval < 0.01:
        p = 0.01
        stars="**"
    elif pval < 0.05:
        p = 0.05
        stars="*"
    else:
        p = 0.1
        stars=''
    return p, stars

# def discrete_cmap(N, base_cmap=None):
#     """Create an N-bin discrete colormap from the specified input map"""

#     # Note that if base_cmap is a string or None, you can simply do
#     #    return plt.cm.get_cmap(base_cmap, N)
#     # The following works for string, None, or a colormap instance:

#     base = plt.cm.get_cmap(base_cmap)
#     color_list = base(np.linspace(0, 1, N))
#     cmap_name = base.name + str(N)
#     return ListedColormap(cmap_name, color_list, N)
def plot_roi_map(values, rois, subject, subjects_dir, cmap="plasma",hemi='both', alpha=1.0):
    
   
    cmap = mpl.cm.viridis_r
    cmap = [cmap(i) for i in np.linspace(0, 1, 4)]
    dic={'0.1':0,'0.05':1,'0.01':2,'0.001':3,}

    brain = Brain(
        subject=subject, subjects_dir=subjects_dir, surf="inflated", hemi=hemi
    )
    labels_lh = np.zeros(len(brain.geo["lh"].coords), dtype=int)
    labels_rh = np.zeros(len(brain.geo["rh"].coords), dtype=int)
    ctab_lh = list()
    ctab_rh = list()
    for i, (roi, value) in enumerate(zip(rois, values), 1):
        if roi.hemi == "lh":
            labels = labels_lh
            ctab = ctab_lh
        else:
            labels = labels_rh
            ctab = ctab_rh
        labels[roi.vertices] = i
        
        ctab.append([int(x * 255) for x in cmap[dic[value]][:4]] + [i])
    ctab_lh = np.array(ctab_lh)
    ctab_rh = np.array(ctab_rh)
    brain.add_annotation(
        [(labels_lh, ctab_lh), (labels_rh, ctab_rh)], borders=False, alpha=alpha
    )
    return brain
# def create_functional_aparc_ROIs(stcs, thre=0.6):
#     labels = mne.read_labels_from_annot('fsaverage', 'aparc', 'both',
#                                         )
#     func_rois = []
#     for roiname in roinames_time:

#         if roiname == 'STC':
#             labels_name = ['superiortemporal-lh',
#                            'bankssts-lh', 'transversetemporal-lh']
#         elif roiname == 'OTC':
#             labels_name = ['fusiform-lh']
#         elif roiname == 'precentral':
#             labels_name = ['precentral-lh']
#         elif roiname == 'IFG':
#             labels_name = ['parsorbitalis-lh', 'parstriangularis-lh', 'parsopercularis-lh',
#                            'lateralorbitofrontal-lh', 'frontalpole-lh']

#         tmin, tmax = roinames_time[roiname][:-1]
#         # stc = stcs[roinames_time[roiname][-1]]-stcs[0]
#         stc = np.mean(stcs)

#         for j in np.arange(0, len(labels_name)):
#             l = [label for label in labels if label.name ==
#                  labels_name[j]][0]
#             if j == 0:
#                 label = l
#             else:
#                 label += l

#         stc_mean = stc.copy().crop(tmin, tmax).mean()
#         stc_mean_label = stc_mean.in_label(label)
#         data = np.abs(stc_mean_label.data)
#         stc_mean_label.data[data < thre * np.max(data)] = 0.0
#         func_labels, _ = mne.stc_to_label(
#             stc_mean_label,
#             src=mne.read_source_spaces(fname.fsaverage_src),
#             smooth=True,
#             connected=True,
#             verbose="error",
#         )

#         func_rois.append(func_labels[0])

#     return func_rois
if __name__ == "__main__":
    # stcs=[]
    # mne.set_config('SUBJECTS_DIR', fname.mri_subjects_dir)
    # SUBJECT = 'fsaverage'
    # for i, cat in enumerate(event_id):
    #     stc = mne.read_source_estimate(fname.ga_stc1(category=cat))
    #     stcs.append(stc)
    #     stc.subject=SUBJECT
 
    # func_rois=create_grow_ROIs1(stcs,extents=15,plot=True)
    labels=select_rois(rois_id=[34,32,50,62,])
    