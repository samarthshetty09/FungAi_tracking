#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 00:11:43 2024

@author: samarth
"""
import os
import numpy as np
import scipy.io as sio
from skimage.morphology import thin
from skimage.filters import threshold_otsu
from shutil import copyfile

pos = 'Pos0_2'
path = os.path.join('E:', 'SR_Tracking', 'toy_data', pos, '')
sav_path = os.path.join('E:', 'SR_Tracking', 'toy_data', 'Tracks', '')

# Removes the mating events from sposeg tracks based on the overlapping tracked indices 

mat_track_path = os.path.join(sav_path, f'{pos}_MAT_16_18_Track1')
if any(os.path.isfile(os.path.join(mat_track_path, f)) for f in os.listdir(mat_track_path)):
    file_list = [f for f in os.listdir(mat_track_path) if os.path.isfile(os.path.join(mat_track_path, f))]

    mat = sio.loadmat(os.path.join(sav_path, file_list[0]))

    art_track_path = os.path.join(sav_path, f'{pos}_ART_Track')
    if any(os.path.isfile(os.path.join(art_track_path, f)) for f in os.listdir(art_track_path)):
        file_list = [f for f in os.listdir(art_track_path) if os.path.isfile(os.path.join(art_track_path, f))]
        art = sio.loadmat(os.path.join(sav_path, file_list[0]))

        Mask3 = art['Mask3']
        Matmasks = mat['Matmasks']

        for iv in range(mat['no_obj'][0, 0]):
            indx_remov = []
            final_indx_remov = []
            for its in range(mat['cell_data'][iv, 0], mat['cell_data'][iv, 1] + 1):  # check for 10 time points
                M = Matmasks[0, its]
                M0 = (M == iv + 1).astype(np.uint16)
                A = Mask3[0, its]
                M1 = M0 > threshold_otsu(M0)
                M2 = thin(M1, max_iter=30)
                M3 = A * M2
                indx = np.unique(A[M3 != 0])
                if indx.size > 0:
                    for itt2 in indx:
                        if np.sum(M3 == itt2) > 5:
                            indx_remov.append(itt2)

            if len(indx_remov) > 0:
                indx_remov_inter = np.unique(indx_remov)
                final_indx_remov = np.unique(indx_remov)
                for itt1 in indx_remov_inter:
                    dist_data = -1 * np.ones(len(Mask3))
                    for its1 in range(mat['cell_data'][iv, 0], art['cell_exists'][1, itt1] + 1):
                        if its1 >= art['cell_exists'][0, itt1]:
                            M6 = (Mask3[0, its1] == itt1)
                            M7 = (Matmasks[0, its1] == iv + 1)
                            dist_data[its1] = np.sum(M6 * M7) / np.sum(M6)
                    
                    if np.any(dist_data != -1):
                        first_ov = np.where(dist_data != -1)[0][0]
                        last_ov = np.where(dist_data != -1)[0][-1]
                        val_avg = np.median(dist_data[first_ov:last_ov])
                        if val_avg <= 0.4:
                            final_indx_remov = np.setdiff1d(final_indx_remov, itt1)
                
                for its in range(mat['cell_data'][iv, 0], len(Mask3)):
                    for itt in final_indx_remov:
                        Mask3[0, its][Mask3[0, its] == itt] = 0
            print(iv)
        
        shock_period = mat['shock_period']
        no_obj = art['no_obj']
        ccell2 = art['ccell2']
        cell_exists = art['cell_exists']
        im_no = art['im_no']
    else:
        art_track_path = os.path.join(sav_path, f'{pos}_ART_Track')
        file_list = [f for f in os.listdir(art_track_path) if os.path.isfile(os.path.join(art_track_path, f))]
        filename_old = file_list[0]
        filename_new = filename_old.replace('_ART_Track', '_ART_Track1')
        copyfile(os.path.join(sav_path, filename_old), os.path.join(sav_path, filename_new))

    sio.savemat(os.path.join(sav_path, f'{pos}_ART_Track1.mat'), {
        "no_obj": no_obj,
        "shock_period": shock_period,
        "Mask3": Mask3,
        "im_no": im_no,
        "ccell2": ccell2,
        "cell_exists": cell_exists
    }, do_compression=True)

