import os
import numpy as np
import h5py
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage import img_as_ubyte
from skimage.io import imshow

# Initialize variables as needed
pos = 'Pos0_2'
path = '/Users/samarth/Documents/MirandaLabs/tracking_algo/FungAi_tracking/Tracking_toydata_Tracks'  # Path to segment SpoSeg masks
sav_path = '/Users/samarth/Documents/MirandaLabs/tracking_algo/FungAi_tracking/Tracking_toydata_Tracks'  # Path to save Tracks

def load_mat(filename):
    try:
        return sio.loadmat(filename)
    except NotImplementedError:
        # Load using h5py for MATLAB v7.3 files
        data = {}
        with h5py.File(filename, 'r') as f:
            for key in f.keys():
                data[key] = np.array(f[key])
        return data

# Load ART and TET tracks
art_file_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
art = load_mat(os.path.join(path, art_file_list[2]))  # load art track

tet_path = os.path.join(path, f'{pos}_TET_Track_DS')
tet_file_list = [f for f in os.listdir(tet_path) if os.path.isfile(os.path.join(tet_path, f))]
tet = load_mat(os.path.join(tet_path, tet_file_list[0]))  # load tet track

print("Keys in ART:", art.keys())
print("Keys in TET:", tet.keys())

# Extract necessary variables from loaded data
Mask3 = art['Mask3']
shock_period = art['shock_period']
TET_obj = int(tet['TET_obj'][0, 0])
TET_exists = tet['TET_exists']
TETmasks = tet['TETmasks']

TET_ID = np.zeros((1, TET_obj))

print(shock_period[1,0])

for iv in range(TET_obj):
    its = int(TET_exists[iv, 0])
    
    if its > shock_period[1, 0]:
        TET_ID[0, iv] = -1
    else:
        A = Mask3[its,:, :].astype(float)
        plt.figure()
        plt.imshow(A, cmap='gray')
        plt.title('A')
        plt.show()

        T = resize(TETmasks[its, :, :], A.shape, order=0, preserve_range=True).astype(float)
        """
        plt.figure()
        plt.imshow(T, cmap='gray')
        plt.title('T')
        plt.show()
        """
        T1 = (T == iv + 1).astype(float)
        """
        plt.figure()
        plt.imshow(T1, cmap='gray')
        plt.title('T1')
        plt.show()
        """

        Im1 = (T1 > threshold_otsu(T1)).astype(float)
        """
        plt.figure()
        plt.imshow(Im1, cmap='gray')
        plt.title('Im1')
        plt.show()
        """

        Im2 = binary_erosion(Im1, np.ones((9, 9))).astype(float)
        """
        plt.figure()
        plt.imshow(Im2, cmap='gray')
        plt.title('Im2')
        plt.show()
        """

        Im3 = A * Im2
        """
        plt.figure()
        plt.imshow(Im3, cmap='gray')
        plt.title('Im3')
        plt.show()
        """

        pix1 = np.unique(A[Im3 != 0])
        TET_ID[0, iv] = pix1[0] if pix1.size > 0 else -1

name1 = os.path.join(sav_path, f'{pos}_TET_ID_art_track.mat')
sio.savemat(name1, {'TET_ID': TET_ID})
