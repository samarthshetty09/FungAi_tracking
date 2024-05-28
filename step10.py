import os
import numpy as np
import scipy.io as sio
from skimage.morphology import binary_erosion
from skimage.transform import resize
from skimage.filters import threshold_otsu

# Initialize variables
pos = 'Pos0_2'
path = os.path.join('E:', 'SR_Tracking', 'toy_data', pos, '')
sav_path = os.path.join('E:', 'SR_Tracking', 'toy_data', 'Tracks', '')

# Load ART and TET tracks
art_path = os.path.join(sav_path, f'{pos}_ART_Track_DS')
art_file_list = [f for f in os.listdir(art_path) if os.path.isfile(os.path.join(art_path, f))]
art = sio.loadmat(os.path.join(sav_path, art_file_list[0]))

tet_path = os.path.join(sav_path, f'{pos}_TET_Track_DS')
tet_file_list = [f for f in os.listdir(tet_path) if os.path.isfile(os.path.join(tet_path, f))]
tet = sio.loadmat(os.path.join(sav_path, tet_file_list[0]))

# Finding the corresponding cell number of TET cells (in TET tracks) in ART tracks
TET_ID = np.zeros((1, tet['TET_obj'][0, 0]))

for iv in range(tet['TET_obj'][0, 0]):
    its = tet['TET_exists'][iv, 0]

    if its > art['shock_period'][0, 1]:
        TET_ID[0, iv] = -1
    else:
        A = art['Mask3'][0, its].astype(float)
        T = resize(tet['TETmasks'][0, its], A.shape, order=0, preserve_range=True).astype(float)
        
        T1 = (T == iv + 1).astype(float)
        Im1 = (T1 > threshold_otsu(T1)).astype(float)
        Im2 = binary_erosion(Im1, np.ones((9, 9))).astype(float)
        Im3 = A * Im2
        
        pix1 = np.unique(A[Im3 != 0])
        TET_ID[0, iv] = pix1[0] if pix1.size > 0 else -1

name1 = os.path.join(sav_path, f'{pos}_TET_ID_art_track.mat')
sio.savemat(name1, {'TET_ID': TET_ID})
