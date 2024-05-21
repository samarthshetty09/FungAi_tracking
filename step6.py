# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:02:08 2024

@author: samar
"""

import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.morphology import thin, remove_small_objects, binary_fill_holes
import scipy.io as sio
from skimage.measure import regionprops, label

# Define paths and parameters
pos = 'Pos0_2'
path = f'E:\\SR_Tracking\\toy_data\\{pos}\\'
sav_path = 'E:\\SR_Tracking\\toy_data\\Tracks\\'

mat_track_path = os.path.join(sav_path, f'{pos}_MAT_16_18_Track')
file_list = os.listdir(mat_track_path)
mat = sio.loadmat(os.path.join(sav_path, file_list[0]))  # Load mat tracks

if mat['no_obj'][0][0] != 0:  # number of positive mat detections
    shock_period = mat['shock_period']
    MTrack = mat['Matmasks']
    no_obj = mat['no_obj'][0][0]
    cell_data = mat['cell_data']

    art_track_path = os.path.join(sav_path, f'{pos}_ART_Track')
    file_list = os.listdir(art_track_path)
    art = sio.loadmat(os.path.join(sav_path, file_list[0]))
    art_masks = art['Mask3']
    mat_artifacts = []

    for its in range(len(MTrack)):
        if MTrack[its].size > 0:
            MTrack[its] = resize(MTrack[its], art_masks[its].shape, order=0, preserve_range=True).astype(np.uint16)

    tp_end = len(art_masks)
    if len(MTrack) != tp_end:
        for its in range(len(MTrack), tp_end):
            MTrack.append(np.zeros_like(MTrack[min(cell_data[:, 0])]))

    # In MatSeg Tracks correction, mating events whose variance of the eccentricity
    # of each mask across the total valid time points is greater than 0.02 are artifacts
    cor_data = np.zeros((3, no_obj))
    size_cell = np.zeros((no_obj, len(MTrack)))
    morph_data = np.zeros((no_obj, len(MTrack)))
    outlier_tps = [None] * no_obj
    good_tps = [None] * no_obj

    for iv in range(no_obj):
        intv = range(cell_data[iv, 0], cell_data[iv, 1] + 1)
        for its in intv:
            M = (MTrack[its] == iv).astype(np.uint8)
            size_cell[iv, its] = np.sum(M)
            val = regionprops(M, 'eccentricity')
            morph_data[iv, its] = val[0].eccentricity
        cor_data[0, iv] = np.mean(size_cell[iv, intv])  # average
        cor_data[1, iv] = np.std(size_cell[iv, intv])  # standard deviation
        cor_data[2, iv] = cor_data[1, iv]  # threshold
        outlier_tps[iv] = [i for i in intv if abs(size_cell[iv, i] - cor_data[0, iv]) > cor_data[2, iv]]
        good_tps[iv] = list(set(intv) - set(outlier_tps[iv]))

    for iv in range(no_obj):
        intv = range(cell_data[iv, 0], cell_data[iv, 1] + 1)
        if np.var(morph_data[iv, intv]) > 0.02:
            mat_artifacts.append(iv)

    for iv in range(no_obj):
        outlier = sorted(outlier_tps[iv])
        good = sorted(good_tps[iv])
        intv = range(cell_data[iv, 0], cell_data[iv, 1] + 1)
        while outlier:
            its = min(outlier)
            gtp = its if its == min(intv) else max([g for g in good if g < its], default=min([g for g in good if g > its]))
            A = art_masks[its].astype(np.uint16)
            M1 = (MTrack[gtp] == iv).astype(np.uint8)
            M2 = thin(M1, max_iter=30).astype(np.uint8)
            M3 = (A * M2).astype(np.uint16)
            indx = np.unique(A[M3 != 0])
            if indx.size > 0:
                X1 = np.zeros_like(MTrack[its], dtype=np.uint8)
                for itt2 in indx:
                    if np.sum(M3 == itt2) > 5:
                        X1[A == itt2] = 1
                X1 = binary_fill_holes(X1).astype(np.uint8)
                X2 = label(X1)
                if X2.max() > 1:
                    continue
                if abs(np.sum(X1) - cor_data[0, iv]) > 2 * cor_data[1, iv]:
                    MTrack[its][MTrack[its] == iv] = 0
                    MTrack[its][MTrack[gtp] == iv] = iv
                else:
                    MTrack[its][MTrack[its] == iv] = 0
                    MTrack[its][X1 == 1] = iv
            outlier.remove(its)
            good.append(its)

    for iv in range(no_obj):
        if cell_data[iv, 1] != tp_end:
            count = 0
            for its in range(cell_data[iv, 1] + 1, tp_end):
                A = art_masks[its]
                M1 = (MTrack[its - 1] == iv).astype(np.uint8)
                M2 = thin(M1, max_iter=30).astype(np.uint8)
                M3 = (A * M2).astype(np.uint16)
                indx = np.unique(A[M3 != 0])
                if indx.size > 0:
                    X1 = np.zeros_like(MTrack[its], dtype=np.uint8)
                    for itt2 in indx:
                        if np.sum(M3 == itt2) > 5:
                            X1[A == itt2] = 1
                    if abs(np.sum(X1) - cor_data[0, iv]) > 2 * cor_data[1, iv]:
                        count += 1
                        MTrack[its][MTrack[its - 1] == iv] = iv
                    else:
                        MTrack[its][X1 == 1] = iv
                else:
                    count += 1
                    MTrack[its][MTrack[its - 1] == iv] = iv
            if count / len(range(cell_data[iv, 1], tp_end)) > 0.8:
                mat_artifacts.append(iv)

    # Remove cell artifacts and rename
    if mat_artifacts:
        all_ccel = list(range(1, no_obj + 1))
        mat_artifacts = np.unique(mat_artifacts)
        for iv in mat_artifacts:
            for its in range(len(MTrack)):
                MTrack[its][MTrack[its] == iv] = 0

        good_cells = sorted(set(all_ccel) - set(mat_artifacts))

        for iv in good_cells:
            for its in range(len(MTrack)):
                MTrack[its][MTrack[its] == iv] = iv

        no_obj = len(good_cells)

    # Recalculating MAT Data
    all_obj_new = SR_240222_cal_allob(no_obj, MTrack, range(len(MTrack)))
    cell_data_new = SR_240222_cal_celldata(all_obj_new, no_obj)

    cell_data = cell_data_new
    all_obj = all_obj_new
    Matmasks = MTrack

    sio.savemat(os.path.join(sav_path, f'{pos}_MAT_16_18_Track1.mat'), {
        "Matmasks": Matmasks,
        "all_obj": all_obj,
        "cell_data": cell_data,
        "no_obj": no_obj,
        "shock_period": shock_period,
        "mat_artifacts": mat_artifacts
    })

