#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 20:23:21 2023

@author: jiaxin
"""
# %% import modules
import pandas as pd
import re
import numpy as np
#%%
log_folder="/m/nbe/archive/flexwordrec/experiment/logs"
# %% block 1
subject = 4
subjects = []
for subject in range(27,28):

    log_file = pd.read_csv(f'{log_folder}/sub-{subject:02}-flexwordrec.log',
                           skiprows=3,
                           # nrows=2000,
                           sep='\t', encoding='utf-8')[1:]
    # index of last row
    last_ind = log_file[log_file['Code'] == 'pause'].index[-1]
    log_file = log_file.iloc[:last_ind]
    # log_file = pd.read_csv(f'/m/nbe/archive/flexwordrec/Experiment/logs/pilot3-flexwordrec.log',
    #                        skiprows=3, nrows=1909, sep='\t', encoding='latin1')[1:]
    # %%% stimuli number and duration
    n_stimuli = len(log_file[log_file['Code'] == 'fix'])
    duration = (log_file['Time'].iloc[-1]-log_file['Time'].iloc[2])/(10000*60)
    print(
        'Subject:', f'sub-{subject:02}'
        # 'Subject:', subject,
        # 'Number of stimuli: ', n_stimuli,
        '\n', 'Duration (min): ', round(duration, 3))

    # %%% behaviriol results
    d = {"hit": [], "incorrect": [], "miss": [], "Accuracy": [], "RT(s)": []}
    d_all = {"hit": [], "incorrect": [],
             "miss": [], "Accuracy": [], "RT": []}

    for i in range(1, 5):
        regexp = re.compile(r"("+str(i)+"_[0,1]-)")
        catch = log_file[log_file['Code'].str.match(regexp, na=False)]
        if len(catch) > 15:
            catch = catch.iloc[-15:]
        ind = catch.index.tolist()
        perf = log_file['Stim Type'][ind]
        sen = log_file['Code'][ind]
        n_hit = perf.value_counts()['hit'] if 'hit' in list(perf) else 0
        d['hit'].append(n_hit)

        n_miss = perf.value_counts()['miss'] if 'miss' in list(perf) else 0
        d['miss'].append(n_miss)

        n_inc = perf.value_counts(
        )['incorrect'] if 'incorrect' in list(perf) else 0
        d['incorrect'].append(n_inc)

        if n_inc+n_hit+n_miss == 0:
            d['Accuracy'].append('N/A')
        else:
            d['Accuracy'].append(round(n_hit/(n_inc+n_hit+n_miss), 3))

        # reaction time
        rea_tim = []
        ind_rep = [i+1 for i in ind]
        for j in ind_rep:
            if perf[j-1] == 'hit':
                rea_tim.append(log_file['TTime'][j]/10000)

            # if log_file['Event Type'][j] == 'Response':
            #     rea_tim.append(log_file['TTime'][j]/10000)
            # else:
            #     rea_tim.append(3)
        d['RT(s)'].append(round(np.mean(rea_tim), 3))

    df = pd.DataFrame(d, index=["RW", "RL1PW", 'RL2PW', 'RL3PW'])
    print(df, '\n-------------------------------------------------------')
    # perfs = {"RW": [list((df['hit'][0], df['incorrect'][0], df['miss'][0]))],
    #          "RL1PW": [list((df['hit'][1], df['incorrect'][1], df['miss'][1]))],
    #          "RL2PW": [list((df['hit'][2], df['incorrect'][2], df['miss'][2]))],
    #          "RL3PW": [list((df['hit'][3], df['incorrect'][3], df['miss'][3]))], }
    perfs = {"RW": str(df['hit'][0])+","+str(df['incorrect'][0])+","+str(df['miss'][0]),
              "RL1PW": str(df['hit'][1])+","+str(df['incorrect'][1])+","+str(df['miss'][1]),
              "RL2PW": str(df['hit'][2])+","+str(df['incorrect'][2])+","+str(df['miss'][2]),
              "RL3PW": str(df['hit'][3])+","+str(df['incorrect'][3])+","+str(df['miss'][3]), }
    df = pd.DataFrame(perfs, index=[f'sub-{subject:02}'],)
    if subject == 1:
        df.to_csv('behave_result/perfs.csv')
    else:
        df.to_csv('behave_result/perfs.csv', mode='a', header=False)
    # # %%
    Accs = {"RW": d['Accuracy'][0], "RL1PW": d['Accuracy'][1],
            "RL2PW": d['Accuracy'][2],  "RL3PW": d['Accuracy'][3]}
    df = pd.DataFrame(Accs, index=[f'sub-{subject:02}'])
    # %%
    if subject == 1:
        df.to_csv('behave_result/accs.csv')
    else:
        df.to_csv('behave_result/accs.csv', mode='a', header=False)

    # %%
    rts = {"RW": d['RT(s)'][0], "RL1PW": d['RT(s)'][1],
           "RL2PW": d['RT(s)'][2],  "RL3PW": d['RT(s)'][3]}
    df = pd.DataFrame(rts, index=[f'sub-{subject:02}'])
    if subject == 1:
        df.to_csv('behave_result/rts.csv')
    else:
        df.to_csv('behave_result/rts.csv', mode='a', header=False)

    # %%perf
   
