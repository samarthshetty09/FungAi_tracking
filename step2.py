# -*- coding: utf-8 -*-
"""
Created on Mon May 20 18:23:38 2024

@author: samar
"""

import os
import numpy as np
import scipy.io as sio
from skimage.io import imread
from skimage.morphology import skeletonize
from scipy.stats import mode
from functions.SR_240222_cal_allob import SR_240222_cal_allob
from functions.SR_240222_cal_celldata import SR_240222_cal_celldata 

# Define thresholds and parameters
thresh_percent = 0.015
thresh_remove_last_mask = 10
thresh_next_cell = 400
thresh = 80
shock_period = [122, 134]

pos = 'Pos0_2'
path = f'E:\\SR_Tracking\\toy_data\\{pos}\\'
sav_path = 'E:\\SR_Tracking\\toy_data\\Tracks\\'

# Load image file names
file_names = [f for f in os.listdir(path) if f.endswith('_Ph3_000_TET_masks.tif')]
file_numbers = [int(f.split('img_')[-1].split('.')[0]) for f in file_names]

sorted_indices = np.argsort(file_numbers)
sorted_numbers = np.array(file_numbers)[sorted_indices]
tet_masks_path = [os.path.join(path, file_names[i]) for i in sorted_indices]

# Read images
tet_masks = [imread(tet_masks_path[i]) for i in range(len(file_names))]
tet_masks = [tet_masks[i] if i in sorted_numbers else np.zeros_like(tet_masks[0], dtype=np.uint16) for i in range(max(sorted_numbers) + 1)]

# Remove shock induced timepoints
if shock_period:
    for start, end in shock_period:
        for i in range(start, end + 1):
            tet_masks[i] = None

# Find the first non-empty mask
start = next((i for i, mask in enumerate(tet_masks) if mask is not None and np.sum(mask) > 0), 0)

# Tracking all the detections
if start != 0:
    rang = range(start, len(tet_masks))
    I2 = tet_masks[start]
    A = np.zeros_like(I2)
else:
    rang = range(len(tet_masks))
    I2 = tet_masks[0]
    A = np.zeros_like(I2)

IS6 = np.zeros_like(I2)
TETC = [np.zeros_like(I2) for _ in range(2 * len(tet_masks))]
xx = start
ccel = 1

while xx != 0:
    k = 0
    for im_no in rang:
        I2 = tet_masks[im_no] if ccel == 1 else TETC[2 * im_no]
        if I2 is None:
            continue
        
        I3A = IS6 if im_no != min(rang) else (I2 == np.unique(I2)[1])
        I3A = skeletonize(I3A)
        I3B = I3A * I2
        ind = mode(I3B[I3B != 0], axis=None)[0][0]

        if ind == 0 and ccel == 1:
            k += 1
            if k > thresh_next_cell:
                for im_no_1 in range(im_no, rang[-1] + 1):
                    if tet_masks[im_no_1] is not None:
                        TETC[2 * im_no_1] = tet_masks[im_no_1]
                break
            else:
                TETC[2 * im_no] = I2
                continue
        elif ind == 0 and ccel != 1:
            k += 1
            if k > thresh_next_cell:
                break
            else:
                continue
        
        k = 0
        pix = np.where(I2 == ind)
        I2[pix] = ccel
        IS6 = I2
        TETC[2 * im_no] = I2
        TETC[2 * im_no + 1] = I2

    xx = next((i for i in rang if TETC[2 * i] is not None and np.sum(TETC[2 * i]) > 0), 0)
    ccel += 1

ccel -= 1

# Removing the shock induced points from rang
rang3 = list(rang)
if shock_period:
    for start, end in shock_period:
        rang3 = [i for i in rang3 if i < start or i > end]

# Removing artifacts
all_obj = SR_240222_cal_allob(ccel, TETC, rang)
cell_data = SR_240222_cal_celldata(all_obj, ccel)

cell_artifacts = [iv for iv in range(ccel) if cell_data[iv, 3] < thresh_percent * len(rang3) or cell_data[iv, 5] > thresh]
all_ccel = range(1, ccel + 1)

if cell_artifacts:
    for iv in cell_artifacts:
        for its in rang3:
            pix = np.where(TETC[2 * its] == iv)
            TETC[2 * its][pix] = 0

# Retaining and relabeling the new cells
good_cells = sorted(set(all_ccel) - set(cell_artifacts))

for iv in good_cells:
    for its in rang3:
        pix = np.where(TETC[2 * its] == iv)
        TETC[2 * its][pix] = iv

# Correcting the SpoSeg track masks or filling the empty spaces
all_obj1 = SR_240222_cal_allob(len(good_cells), TETC, rang)

cell_data1 = SR_240222_cal_celldata(all_obj1, len(good_cells))

for iv in good_cells:
    for its in range(cell_data1[iv, 0] + 1, cell_data1[iv, 1]):
        if all_obj1[iv, its] == 0:
            prev = next(i for i in range(cell_data1[iv, 0], its) if all_obj1[iv, i] > 0)
            all_obj1[iv, its] = all_obj1[iv, prev]
            pix = np.where(TETC[2 * prev] == iv)
            TETC[2 * its][pix] = iv

# Cell array that contains the fully tracked TetSeg masks
TETmasks = [TETC[2 * ita] for ita in range(len(TETC) // 2)]

# Calculate the size of tetrads
TET_obj = len(good_cells)
all_obj_final = SR_240222_cal_allob(TET_obj, TETmasks, range(len(TETmasks)))

TET_Size = all_obj_final

# Calculate first detection and last detection of tetrads
TET_exists = np.zeros((2, TET_obj), dtype=int)
for iv in range(TET_obj):
    TET_exists[0, iv] = next(i for i in range(len(TET_Size[iv])) if TET_Size[iv, i] > 0)
    TET_exists[1, iv] = next(i for i in reversed(range(len(TET_Size[iv]))) if TET_Size[iv, i] > 0)

# Save the results
sio.savemat(os.path.join(sav_path, f'{pos}_TET_Track.mat'), {
    'start': start,
    'TET_Size': TET_Size,
    'TET_obj': TET_obj,
    'TET_exists': TET_exists,
    'TETmasks': TETmasks,
    'shock_period': shock_period,
    'thresh': thresh,
    'thresh_next_cell': thresh_next_cell,
    'thresh_percent': thresh_percent,
    'tet_masks_exists_tp': rang3
}, do_compression=True)
