# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:03:27 2024

@author: samar
"""

import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.morphology import binary_erosion, disk
from skimage.filters import threshold_otsu
import scipy.io as sio

# Define paths and parameters
pos = 'Pos0_2'
path = f'E:\\SR_Tracking\\toy_data\\{pos}\\'
sav_path = 'E:\\SR_Tracking\\toy_data\\Tracks\\'
shock_period = [122, 134]

# Load ART masks
path_dir = [f for f in os.listdir(path) if f.endswith('_ART_masks.tif')]
Art_MT = [imread(os.path.join(path, f)).astype(np.uint16) for f in path_dir]

# Load tracked SpoSeg masks
tet_track_path = os.path.join(sav_path, f'{pos}_TET_Track.mat')
if os.path.exists(tet_track_path):
    tet = sio.loadmat(tet_track_path)
    shock_period = tet['shock_period']

    for iv in range(tet['TET_obj'][0][0]):
        if tet['TET_exists'][1, iv] >= shock_period[0] - 1:
            tp_end = shock_period[1]
        else:
            tp_end = tet['TET_exists'][1, iv]

        for its in range(tet['TET_exists'][0, iv], tp_end + 1):
            A1 = Art_MT[its].astype(np.double)
            if shock_period[0] <= its <= shock_period[1]:
                T1 = (tet['TETmasks'][0, shock_period[0] - 1] == iv).astype(np.double)
                thresh = 0.6
            else:
                T1 = (tet['TETmasks'][0, its] == iv).astype(np.double)
                thresh = 0.95

            T1 = resize(T1, A1.shape, order=0, preserve_range=True)
            Im1 = T1 > threshold_otsu(T1)
            Im2 = binary_erosion(Im1, selem=disk(9))
            Im3 = A1 * Im2

            pix11 = []
            pix1 = np.unique(A1[Im3 != 0])
            for it2 in pix1:
                r1 = np.sum(Im3 == it2) / np.sum(Im3 > 0)
                if r1 > 0.2:
                    pix11.append(it2)

            if len(pix11) == 1:
                r = np.sum(A1 == pix11[0]) / np.sum(T1)
                if r > thresh:
                    pass
                else:
                    Art_MT[its][A1 == pix11[0]] = 0
                    Art_MT[its][T1 == 1] = np.max(Art_MT[its]) + 1
            elif not pix11:
                Art_MT[its][T1 == 1] = np.max(Art_MT[its]) + 1
            else:
                for it2 in pix11:
                    Art_MT[its][A1 == it2] = 0
                Art_MT[its][T1 == 1] = np.max(Art_MT[its]) + 1

    for iv in range(tet['TET_obj'][0][0]):
        if tet['TET_exists'][1, iv] > shock_period[1] and tet['TET_exists'][0, iv] < shock_period[0]:
            s1 = np.sum(tet['TETmasks'][0, shock_period[1]] == iv)
            for its in range(shock_period[1], tet['TET_exists'][1, iv] + 1):
                A1 = Art_MT[its].astype(np.double)
                T1 = (tet['TETmasks'][0, its] == iv).astype(np.double)

                s2 = np.sum(tet['TETmasks'][0, its] == iv)
                if its == tet['TET_exists'][1, iv]:
                    s3 = np.sum(tet['TETmasks'][0, its] == iv)
                else:
                    s3 = np.sum(tet['TETmasks'][0, its + 1] == iv)

                if s2 < s1 - 0.1 * s1:
                    if s3 > s2 + 0.1 * s2:
                        T1 = (tet['TETmasks'][0, its - 1] == iv).astype(np.double)
                    else:
                        break

                s1 = s2
                T1 = resize(T1, A1.shape, order=0, preserve_range=True)
                Im1 = T1 > threshold_otsu(T1)
                Im2 = binary_erosion(Im1, selem=disk(9))
                Im3 = A1 * Im2

                pix11 = []
                pix1 = np.unique(A1[Im3 != 0])
                for it2 in pix1:
                    r1 = np.sum(Im3 == it2) / np.sum(Im3 > 0)
                    if r1 > 0.2:
                        pix11.append(it2)

                if len(pix11) == 1:
                    r = np.sum(A1 == pix11[0]) / np.sum(T1)
                    if r > thresh:
                        pass
                    else:
                        Art_MT[its][A1 == pix11[0]] = 0
                        Art_MT[its][T1 == 1] = np.max(Art_MT[its]) + 1
                elif not pix11:
                    Art_MT[its][T1 == 1] = np.max(Art_MT[its]) + 1
                else:
                    for it2 in pix11:
                        Art_MT[its][A1 == it2] = 0
                    Art_MT[its][T1 == 1] = np.max(Art_MT[its]) + 1

    sio.savemat(os.path.join(sav_path, f'{pos}_ART_Masks.mat'), {"Art_MT": Art_MT, "shock_period": shock_period})
else:
    sio.savemat(os.path.join(sav_path, f'{pos}_ART_Masks.mat'), {"Art_MT": Art_MT, "shock_period": shock_period})
