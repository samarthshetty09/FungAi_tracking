# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:02:00 2024

@author: samarth
"""

import os
import numpy as np
import scipy.io as sio
from skimage.io import imread
from skimage.morphology import skeletonize
from functions.SR_240222_cal_allob import SR_240222_cal_allob
from functions.SR_240222_cal_celldata import SR_240222_cal_celldata 
from scipy.stats import mode

# Define thresholds and parameters
pos = 'Pos0_2'
path = f'E:\\SR_Tracking\\toy_data\\{pos}\\'
sav_path = 'E:\\SR_Tracking\\toy_data\\Tracks\\'
shock_period = [122, 134]

# Load image file names
file_names = [f for f in os.listdir(path) if f.endswith('_Ph3_000_MAT16_18_masks.tif')]
file_numbers = [int(f.split('img_')[1].split('_Ph3_000_MAT16_18_masks.tif')[0]) for f in file_names]

sorted_indices = np.argsort(file_numbers)
sorted_numbers = np.array(file_numbers)[sorted_indices]
mat_masks_path = [os.path.join(path, file_names[i]) for i in sorted_indices]

# Read images
mat_masks = [imread(mat_masks_path[i]) for i in range(len(file_names))]
mat_masks = [mat_masks[i] if i in sorted_numbers else np.zeros_like(mat_masks[0], dtype=np.uint16) for i in range(max(sorted_numbers) + 1)]

# Save all the valid mat masks in a variable called mat_masks
for i in range(min(sorted_numbers) + 1, len(mat_masks)):
    if mat_masks[i] is None:
        mat_masks[i] = np.zeros_like(mat_masks[min(sorted_numbers) + 1], dtype=np.uint16)

# Remove shock induced timepoints
mat_masks_original = mat_masks.copy()

if shock_period:
    for start, end in shock_period:
        for i in range(start, end + 1):
            mat_masks[i] = None

# Find the first non-empty mask
start = next((i for i, mask in enumerate(mat_masks) if mask is not None and np.sum(mask) > 0), 0)

# Tracking all the detections
if start != 0:
    rang = range(start, len(mat_masks))
    I2 = mat_masks[start]
    A = np.zeros_like(I2)
else:
    rang = range(len(mat_masks))
    I2 = mat_masks[0]
    A = np.zeros_like(I2)

IS6 = np.zeros_like(I2)
MATC = [np.zeros_like(I2) for _ in range(2 * len(mat_masks))]
xx = start
ccel = 1

while xx != 0:
    for im_no in rang:
        I2 = mat_masks[im_no] if ccel == 1 else MATC[2 * im_no]
        if I2 is None:
            continue

        if im_no == min(rang):
            ind1 = np.unique(I2)[1:]
            I3A = (I2 == ind1[0])
        else:
            I3A = IS6

        I3A = skeletonize(I3A)
        I2A = I2
        I3B = I3A * I2A
        ind = mode(I3B[I3B != 0], axis=None)[0][0]

        if ind == 0 and ccel == 1:
            MATC[2 * im_no] = I2A
            continue
        elif ind == 0 and ccel != 1:
            continue

        pix = np.where(I2A == ind)
        pix0 = np.where(I2A != ind)
        I2A[pix] = ccel
        I2A[pix0] = 0
        IS6 = I2A
        I22 = np.zeros_like(I2)
        pix1 = np.where(IS6 == ccel)
        I2[pix1] = 0
        pix2 = np.unique(I2)[1:]

        if ccel == 1:
            for ity in pix2:
                pix4 = np.where(I2 == ity)
                I22[pix4] = ity
            MATC[2 * im_no] = IS6
        else:
            if pix2.size > 0:
                for ity in pix2:
                    pix4 = np.where(I2 == ity)
                    I22[pix4] = ity
            else:
                I22 = I2
            IS61 = MATC[2 * im_no]
            IS61[pix] = ccel
            MATC[2 * im_no] = IS61

        MATC[2 * im_no] = I22

    xx = next((i for i in rang if MATC[2 * i] is not None and np.sum(MATC[2 * i]) > 0), 0)
    ccel += 1

ccel -= 1

# Removing the shock induced points from rang
rang3 = list(rang)
if shock_period:
    for start, end in shock_period:
        rang3 = [i for i in rang3 if i < start or i > end]

# Correction Code
all_obj = SR_240222_cal_allob(ccel, MATC[::2], rang)
cell_data = SR_240222_cal_celldata(all_obj, ccel)

for iv in range(ccel):
    if np.any(all_obj[iv, min(rang):shock_period[-1][1]] > 0):
        if all_obj[iv, shock_period[-1][1] + 1] != 0:
            for its in range(shock_period[-1][1] + 1, rang[-1] + 1):
                if all_obj[iv, its] != -1:
                    pix = np.where(MATC[2 * its] == iv)
                    MATC[2 * its][pix] = 0
                    all_obj[iv, its] = np.sum(MATC[2 * its] == iv)

cell_data = SR_240222_cal_celldata(all_obj, ccel)

k = 1
cell_artifacts = []
for iv in range(ccel):
    if cell_data[iv, 2] == 1 or cell_data[iv, 4] > 80:
        cell_artifacts.append(iv)
        k += 1

all_ccel = range(1, ccel + 1)

if cell_artifacts:
    cell_artifacts = np.unique(cell_artifacts)
    for iv in cell_artifacts:
        for its in rang3:
            pix = np.where(MATC[2 * its] == iv)
            MATC[2 * its][pix] = 0

good_cells = sorted(set(all_ccel) - set(cell_artifacts))

for iv in good_cells:
    for its in rang3:
        pix = np.where(MATC[2 * its] == iv)
        MATC[2 * its][pix] = iv

ccel = len(good_cells)
all_obj = SR_240222_cal_allob(ccel, MATC[::2], rang)
cell_data = SR_240222_cal_celldata(all_obj, ccel)

for iv in range(ccel):
    tp_data = {
        iv: {
            1: np.diff(np.where(all_obj[iv, :] > 0)),
            2: np.where(all_obj[iv, :] > 0)
        }
    }
    a = np.where(tp_data[iv][1] > 10)
    if a.size > 0:
        if a[0] == tp_data[iv][1].size - 1:
            pix = np.where(MATC[2 * tp_data[iv][2][a[0] + 1]] == iv)
            MATC[2 * tp_data[iv][2][a[0] + 1]][pix] = 0
        else:
            for its in range(np.where(all_obj[iv, :] > 0)[0][0], tp_data[iv][2][a[0] + 1]):
                pix = np.where(MATC[2 * its] == iv)
                MATC[2 * its][pix] = 0

for iv in range(ccel):
    for its in range(np.where(all_obj[iv, :] > 0)[0][0] + 1, np.where(all_obj[iv, :] > 0)[0][-1]):
        if all_obj[iv, its] == 0:
            prev = np.where(all_obj[iv, :its] > 0)[0][-1]
            all_obj[iv, its] = all_obj[iv, prev]
            pix = np.where(MATC[2 * prev] == iv)
            MATC[2 * its][pix] = iv

all_obj = SR_240222_cal_allob(ccel, MATC[::2], rang)
cell_data = SR_240222_cal_celldata(all_obj, ccel)

no_obj = ccel
Matmasks = [MATC[2 * ita] for ita in rang]

# Save the results
sio.savemat(os.path.join(sav_path, f'{pos}_MAT_16_18_Track.mat'), {
    'Matmasks': Matmasks,
    'no_obj': no_obj,
    'all_obj': all_obj,
    'cell_data': cell_data,
    'rang': rang,
    'rang3': rang3,
    'shock_period': shock_period,
    'mat_masks_original': mat_masks_original,
    'start': start
}, do_compression=True)
