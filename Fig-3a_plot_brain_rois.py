#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import mne
from config import fname,event_id,rois_id,roi_colors,parc
import matplotlib.pyplot as plt
#
SUBJECT='fsaverage'
mne.set_config('SUBJECTS_DIR', fname.mri_subjects_dir)
stcs=[mne.read_source_estimate(fname.ga_stc(category=cat),SUBJECT) for cat in event_id]#(5124,601)->(n_vertices, n_tp)
#%% stc
# func_labels=create_grow_ROIs2(stcs)
# func_labels=create_functional_aparc_ROIs(stcs,roinames_time)
# func_labels=create_aparc_ROIs()
#%%
hemi='lh'
if hemi=='rh':
    rois_id=[i+1 for i in rois_id]
func_labels = []
annotation = mne.read_labels_from_annot(
    'fsaverage', parc=parc,)
rois = [label for label in annotation if 'Unknown' not in label.name]
func_labels=[rois[i] for i in rois_id]
# for roi in rois:
#     label = [label for label in annotation if label.name == (roi+hemi)][0]
#     func_labels.append(label)
#%%
Brain = mne.viz.get_brain_class()
brain = Brain(
    "fsaverage",
    hemi,
    "inflated",
    # 'pial',
    cortex="low_contrast",
    background="white",
    # size=(1200, 600),
    views=['lateral', 'ventral'],
    view_layout='vertical'
)


#%%
#%%
for j, roi in enumerate(func_labels):
    func_label=func_labels[j]
    brain.add_label(func_label,
                    color=roi_colors[j],
                    ) 
    print(j)
brain.show_view()
screenshot = brain.screenshot()
# crop out the white margins
nonwhite_pix = (screenshot != 255).any(-1)
nonwhite_row = nonwhite_pix.any(1)
nonwhite_col = nonwhite_pix.any(0)
cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
fig, ax = plt.subplots(1,1,figsize=(5, 5))
im = ax.imshow(cropped_screenshot)
ax.set_yticks([])
ax.set_xticks([])
ax.set_frame_on(False)
plt.savefig(f"{fname.figures_dir(subject=SUBJECT)}/brain_functional_{hemi}_rois.pdf")


