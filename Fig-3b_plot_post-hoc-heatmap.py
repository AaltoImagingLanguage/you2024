#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:34:19 2023

@author: youj2
"""

# %%
# %matplotlib qt
import numpy as np
import mne
import os
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt

import pandas as pd
# import pingouin as pg
import mne
import matplotlib as mpl
import pingouin as pg
from config import fname, event_id, baseline, subjects, rois_id,rois_names,roi_colors
import utility.figure_setting
from utils import create_grow_ROIs2,create_grow_ROIs1,compare_p,select_rois
from matplotlib.colors import ListedColormap,BoundaryNorm
#%%
cmap = mpl.cm.viridis_r
cmap = [cmap(i) for i in np.linspace(0, 1, 4)]

# cmap = sns.cubehelix_palette(start=0.2, rot=1, 
#                               # light=0.6, 
#                               n_colors=4)
# ListedColormap(cmap)
#%%
cmap_dict = sns.cubehelix_palette(start=-0.5, rot=0.3, light=0.6, n_colors=4)
# cmap_dict = {'cornflowerblue':0.001,'slateblue':0.01,
#           'brown':0.05,"black":0.1,
#          }
ListedColormap(cmap_dict)
#%%
#%
bounds = [0, 0.001, 0.01, 0.05, 0.1]
my_norm = BoundaryNorm(bounds, ncolors=len(cmap_dict))
fontsize=16
# %%
# path to raw data
overwrite=True
SUBJECT = 'fsaverage'
mne.set_config('SUBJECTS_DIR', fname.mri_subjects_dir)
hemi='lh'
# data = np.load(f'{fname.data_dir}/rois_tcs_{hemi}.npy')  # (events,subjects,len(rois), length)
correction='fdr_bh'#'bonf'#fdr_bh,fdr_by,sidak
if hemi=='rh':
    rois_id=[i+1 for i in rois_id]

labels=select_rois(rois_id=rois_id,
                   parc='aparc.a2009s_custom_gyrus_sulcus_800mm2',
                    combines=[]
                   )
# %%

length = 651
src_to = mne.read_source_spaces(fname.fsaverage_src)
X = np.zeros([len(event_id), len(subjects), len(labels), length])

tmin, tmax = baseline


def stc_baseline_correction(ts, stc, tmin, tmax):
    time_dim = len(stc.times)
    # baseline_timepoints = X.times[np.where(X.times<0)]
    # baseline_timepoints = X.times[np.where(X.times==tmin):np.where(X.times==tmax)]
    # Convert tmin/tmax to sample indices
    tmin, tmax = np.searchsorted(stc.times, [tmin, tmax])

    baseline_mean = ts[:, tmin:tmax].mean(1)

    baseline_mean_mat = np.repeat(baseline_mean.reshape([len(baseline_mean), 1]),
                                  time_dim, axis=1)
    corrected_stc = ts - baseline_mean_mat
    return corrected_stc


# %%
for i in np.arange(0, len(subjects)):
    n_subjects = len(subjects)
    subject = subjects[i]
    print('Participant : ', subject)

    # Average the source estimates within each label using sign-flips to reduce
    # signal cancellations, also here we return a generator
    for c, cat in enumerate(event_id):
        stc = mne.read_source_estimate(
            fname.stc_morph(subject=subject, category=cat))
        stc.crop(-0.2, 1.1)
        label_ts = mne.extract_label_time_course(stc, labels, src_to,
                                                 # mode='mean_flip',
                                                   # mode='pca_flip',
                                                  

                                                 )  #

        # label_ts = stc_baseline_correction(label_ts, stc, tmin, tmax)

        X[c, i, :, :] = label_ts

    # for i in np.arange(0,len(labels)-5):

#%%
np.save(f'{fname.data_dir}/rois_tcs_{hemi}', X)
data=X
# %%
def ow_rm_anova(res):

    df1 = res['ddof1'].iloc[0]
    df2 = res['ddof2'].iloc[0]
    F = res['F'].iloc[0].round(2)
    if res['p-unc'].iloc[0] < 0.001:
        p = 0.001
        result = f'F({df1},{df2})={F}, p<{p}'
        stars="***"
    elif res['p-unc'].iloc[0] < 0.01:
        p = 0.01
        stars="**"
        result = f'F({df1},{df2})={F}, p<{p}'
    elif res['p-unc'].iloc[0] < 0.05:
        p = 0.05
        result = f'F({df1},{df2})={F}, p<{p}'
        stars="*"
    else:
        p = res['p-unc'].iloc[0].round(3)
        result = f'F({df1},{df2})={F}, p={p}'
        stars=""
    return result,stars

p_matrix=np.zeros([len(event_id), len(event_id)])
stc = mne.read_source_estimate(fname.ga_stc(category='RW'))
t = stc.times


# times=[times[0]]
# rois=[rois[0]]
times=[[0.3,0.5],[0.5,0.7],[0.7,0.9],[0.9,1.1],
        ]


for r, roi_id in enumerate(labels):
    
        
    f, axs = plt.subplots(1,len(times),figsize=(8, 2),sharey=True,)
    if r==0:
       
       
        cbar_ax = f.add_axes([0.45, 0, .28, .05])
       
        
        
    for t, time in enumerate(times):
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
   
        tmi,tma=time
        tmin, tmax = np.searchsorted(stc.times, [tmi, tma])
    
        data_dict = {'Subject': [], 'tc': [],
                     'Type': [],'condition':[],
                     }
        for j, sub in enumerate(subjects):
            data_dict['Subject'].extend([sub]*4)
            # data_dict['condition'].extend(list(accs_normal.columns))
            data_dict['condition'].extend([0, 1, 2, 3])
            data_dict['Type'].extend(['RW', 'RL1PW', 'RL2PW', 'RL3PW'])
            data_dict['tc'].extend(data[:,j,r,tmin:tmax].mean(-1))
            
        df = pd.DataFrame(data_dict)
        anova=pg.rm_anova(dv='tc', within='condition', subject='Subject',
                                    
                                     data=df)
        
        p_cor=anova['p-unc'][0]
        
        # if p_cor>=0.05:
        # print(r,tmi,tma,p_cor,)
        anova=pg.rm_anova(dv='tc', within='condition', subject='Subject',
                          correction=True,
                                    
                                     data=df)
        p_cor=anova['p-unc'][0]
        result, starts=ow_rm_anova(anova)
        
        # if p_cor>=0.05:
        # print(r,tmi,tma,p_cor,anova['p-spher'][0],anova['p-GG-corr'][0])
        posthocs = pg.pairwise_tests(dv='tc', within='condition', subject='Subject',
                                        # padjust='holm',
                                        # padjust='fdr_bh',
                                        padjust=correction,
                                       # alternative='greater',
                                       # parametric='False',
                                      # return_desc=True,
                                       # effsize='CLES',
                                      data=df)
        
        # print(posthocs)
        p_matrix=np.zeros([len(event_id), len(event_id)])
        for i in range(len(posthocs)):
            rel=posthocs.iloc[i]
            ind0,ind1=np.sort([rel['A'],rel['B']])
            p_value=rel['p-corr']
            # p_value=rel['p-unc']
            # p_value=compare_p(p_value)
            p_matrix[ind1,ind0]=p_value
            
   
        names = [_ for _ in list(event_id.keys())]
        df = pd.DataFrame(p_matrix, index=names, columns=names)
    

       
        #%%
        
        #changed syntax for text, same logic here
        # ticktext = ['<{}'.format(bvals[1])] + ['{} - {}'.format(bvals[k],bvals[k+1]) for k in range(1,len(bvals)-2)] + ['>{}'.format(bvals[-2])]
        mask = np.zeros_like(p_matrix,
                              dtype=bool
                              )
        mask[np.triu_indices_from(mask)] = True
        # mask[np.diag_indices_from(mask)] = False
        # mask = np.triu(df)
            
        # with sns.axes_style("white"):
            # cbar_ax = f.add_axes([0.8, .15, .05, .4])
        df=df.iloc[1:,:-1]
        mask=mask[1:,:-1]
        sns.heatmap(df, 
                          mask=mask,
                            ax=axs[t],
                          # cbar_ax = cbar_ax,
                          square=True, 
                          # linecolor='lightgray',
                          linewidths=0.5,
                            cbar_kws={
                                # 'label': 'p-vaule',
                                "shrink": 0.75,
                                  "location":'bottom',
                                'orientation':'horizontal',
                              
                           
                                },
                            # cmap=ListedColormap(cmap),
                             # cbar_ax=None ,
                             cbar_ax=None if (t & r) else cbar_ax,
                             cbar=(t == 0 & r==0),
                             # cbar=False,
                          cmap=cmap,
                          norm=my_norm,
                          annot=True,
                          annot_kws={"size":8},
                          )
    
            
        if (t==0 & r==0):
            axs[r].set_xticklabels(labels=list(event_id.keys())[:-1],rotation=45,fontsize=fontsize)
            axs[r].set_yticklabels(labels=list(event_id.keys())[1:],rotation=90,fontsize=fontsize)
            colorbar = axs[t].collections[0].colorbar
            colorbar.set_label('p-value',fontsize=fontsize)
            
            
            colorbar.set_ticks([0.001,0.01,0.050])
            colorbar.set_ticklabels(['0.001','0.01','0.05'],fontsize=fontsize)
        else:
            axs[t].set_xticklabels([])
            axs[t].set_yticklabels([])
        tmin1000=int(tmi*1000)
        tmax1000=int(tma*1000)
        
        
        axs[t].set_xticklabels([])
        axs[t].set_yticklabels([])
        if r==len(labels)-1:
            axs[t].set_title(f'{tmin1000}-{tmax1000} ms \n\n{starts}',fontsize=fontsize,
                             y=1.09, 
                               pad=1
                              )
        else:
            axs[t].set_title(f'{starts}',fontsize=fontsize, 
                              # pad=10
                              )
            
       
    print(r,roi_id)
    if r==0:
        axs[r].set_yticklabels(labels=list(event_id.keys())[1:],rotation=360,fontsize=14)
        axs[r].set_xticklabels(labels=list(event_id.keys())[:-1],rotation=45,fontsize=14)
        
        f.suptitle(rois_names[r],fontsize=18,weight='bold',x=0.12,color=roi_colors[r])
      
    elif r==len(labels)-1:
        f.suptitle(rois_names[r],fontsize=18,weight='bold',x=0.12,color=roi_colors[r])
       
    
    else:
        f.suptitle(rois_names[r],fontsize=18,weight='bold',x=0.12,color=roi_colors[r])
    plt.subplots_adjust(wspace=0.5,hspace=0.3)#adjust spaces between subplots
    # colorbar.set_ticks([])
    f.tight_layout(pad=8.0)
    
    f=f'{fname.figures_dir(subject=SUBJECT)}/rois-tc_{hemi}_posthoc_{correction}'
    if not os.path.exists(f):
        os.makedirs(f, exist_ok=True)
    
    plt.savefig(
        f'{f}/roi{r}.pdf',
        bbox_inches='tight'
    )
    # plt.savefig(
    #     f'{f}/roi{r}.png',
    #     bbox_inches='tight'
    # )
  


#%%

