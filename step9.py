#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 00:21:37 2024

@author: samarth
"""

import os
import numpy as np
import scipy.io as sio
from skimage.morphology import thin
from skimage.filters import threshold_otsu
from skimage.measure import label

# Initialize variables
pos = 'Pos0_2'
path = os.path.join('E:', 'SR_Tracking', 'toy_data', pos, '')
sav_path = os.path.join('E:', 'SR_Tracking', 'toy_data', 'Tracks', '')

# Load ART and MAT tracks
art_track_path = os.path.join(sav_path, f'{pos}_ART_Track1_DS')
file_list = [f for f in os.listdir(art_track_path) if os.path.isfile(os.path.join(art_track_path, f))]
art = sio.loadmat(os.path.join(sav_path, file_list[0]))

mat_track_path = os.path.join(sav_path, f'{pos}_MAT_16_18_Track1_DS')
file_list = [f for f in os.listdir(mat_track_path) if os.path.isfile(os.path.join(mat_track_path, f))]
mat = sio.loadmat(os.path.join(sav_path, file_list[0]))

# Obtain the gametes indexes that give rise to the mating cell (zygote)
Mask3 = art['Mask3']  # Reading Art Tracks from art
MTrack = mat['Matmasks']  # Reading Mat Tracks from mat
gamete = np.zeros((3, mat['no_obj'][0, 0]))

for iv in range(mat['no_obj'][0, 0]):
    tp_mat_start = mat['cell_data'][iv, 0]  # First appearance of mating event "iv"
    M1 = MTrack[0, tp_mat_start - 1]  # Mat Track at tp_mat_start
    for its in range(tp_mat_start - 2, mat['shock_period'][0, 1], -1):  # Loop through time from 1 tp before mating to one time point after shock
        A1 = Mask3[0, its].astype(float)
        M2 = (M1 == iv + 1).astype(float)
        
        Im1 = (M2 > threshold_otsu(M2)).astype(float)
        Im2 = thin(Im1, max_iter=10).astype(float)
        Im3 = A1 * Im2
        pix2 = np.unique(A1[Im3 != 0])
        
        if pix2.size == 2:  # captures mature mating
            r = np.sum(Im3 == pix2[0]) / np.sum(Im3 == pix2[1])
            if (2/8) <= r <= (8/2):  # 4/6 to 9/6
                gamete[0, iv] = pix2[0]
                gamete[1, iv] = pix2[1]
                gamete[2, iv] = its + 1

for iv in range(mat['no_obj'][0, 0]):
    if gamete[0, iv] == 0 and gamete[1, iv] == 0:
        tp_mat_start = mat['cell_data'][iv, 0]  # First appearance of mating event "iv"
        M1 = MTrack[0, tp_mat_start - 1]  # Mat Track at tp_mat_start
        for its in range(tp_mat_start - 2, mat['shock_period'][0, 1], -1):  # Loop through time from 1 tp before mating to one time point after shock
            A1 = Mask3[0, its].astype(float)
            M2 = (M1 == iv + 1).astype(float)
            
            Im1 = (M2 > threshold_otsu(M2)).astype(float)
            Im2 = thin(Im1, max_iter=10).astype(float)
            Im3 = A1 * Im2
            pix2 = np.unique(A1[Im3 != 0])
            
            if pix2.size == 1:  # captures ascus mating
                gamete[0, iv] = pix2[0]
                gamete[2, iv] = its + 1

sio.savemat(os.path.join(sav_path, f'{pos}_gametes.mat'), {"gamete": gamete}, do_compression=True)