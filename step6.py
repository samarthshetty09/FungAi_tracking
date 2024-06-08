import os
import numpy as np
import h5py
import scipy.io as sio
from skimage.transform import resize
from skimage.measure import regionprops, label
from skimage.morphology import thin
from scipy.ndimage import binary_fill_holes
from functions.SR_240222_cal_allob import cal_allob
from functions.SR_240222_cal_celldata import cal_celldata
import matplotlib.pyplot as plt

def load_mat_v73(file_path):
    data = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                data[key] = np.array(f[key])
            elif isinstance(f[key], h5py.Group):
                data[key] = {k: np.array(f[key][k]) for k in f[key].keys()}
            elif isinstance(f[key], h5py.h5r.Reference):
                data[key] = [np.array(f[h5py.h5r.dereference(ref, f)]) for ref in f[key]]
    return data


# Modified Cal Allob 
def cal_allob1(ccel, TETC, rang):
    # Initialize the all_obj array with zeros
    all_obj = np.zeros((ccel, len(TETC)))

    for iv in range(0, ccel):  # Adjusted to 1-based index
        for its in rang:
            if TETC[its] is not None:  # Check if the array is not None and not empty
                all_obj[iv, its] = np.sum(TETC[its] == iv)  # Adjusted for 1-based index logic
            else:
                all_obj[iv, its] = -1

    return all_obj

# Define parameters
pos = 'Pos0_2'
path = '/Users/samarth/Documents/MATLAB/Full_Life_Cycle_tracking/tracks/'  # Path to segment SpoSeg masks
sav_path = '/Users/samarth/Documents/MATLAB/Full_Life_Cycle_tracking/saved_res/'  # Path to save Track

# Load MAT Track data
file_list = [f for f in os.listdir(path) if '_MAT_16_18_Track' in f]
mat = load_mat_v73(os.path.join(path, file_list[2]))
Matmasks = mat['Matmasks']

# Extract variables from loaded data
no_obj = int(mat['no_obj'][0])
if no_obj != 0:
    shock_period = mat['shock_period'][0]
    MTrack = Matmasks
    cell_data = mat['cell_data']

    # Load ART Track data
    file_list = [f for f in os.listdir(path) if '_ART_Track' in f]
    art = load_mat_v73(os.path.join(path, file_list[1]))
    art_masks = art["Mask3"]
    mat_artifacts = []

    # Resize MTrack to match ART masks
    for its in range(len(MTrack)):
        if MTrack[its, : , :].size > 1:
            MTrack[its,:,:] = resize(MTrack[its,:,:], art_masks[its,:,:].shape, order=0, preserve_range=True, anti_aliasing=False)

    tp_end = len(art_masks)
    if len(MTrack) != tp_end:
        for its in range(len(MTrack[its,:,:]), tp_end):
            MTrack.append(np.zeros_like(MTrack[int(min(cell_data[:, 0])) - 1,:,:], dtype=np.uint16))

    # Correcting mating tracks
    cor_data = np.zeros((3, no_obj))
    size_cell = np.zeros((no_obj, len(MTrack)))
    morph_data = np.zeros((no_obj, len(MTrack)))
    outlier_tps = [None] * no_obj
    good_tps = [None] * no_obj
    
    for iv in range(no_obj):
        print(iv)

    for iv in range(no_obj):
        int_range = range(int(cell_data[0, iv]) - 1, int(cell_data[1, iv]))  # Adjusting for 0-based indexing
        for its in int_range:
            M = np.uint16(MTrack[its,:,:] == iv)
            
            
            plt.figure()
            plt.imshow(np.uint16(M), cmap='gray')
            plt.title('M')
            plt.show()
            
            
            size_cell[iv, its] = np.sum(M)
            props = regionprops(M)
            morph_data[iv, its] = props[0].eccentricity if props else 0
        cor_data[0, iv] = np.mean(size_cell[iv, int_range])
        cor_data[1, iv] = np.std(size_cell[iv, int_range])
        cor_data[2, iv] = 1 * cor_data[1, iv]
        outlier_tps[iv] = [t for t in int_range if abs(size_cell[iv, t] - cor_data[0, iv]) > cor_data[2, iv]]
        good_tps[iv] = list(set(int_range) - set(outlier_tps[iv]))

    for iv in range(no_obj):
        int_range = range(int(cell_data[0, iv]), int(cell_data[0, iv]))
        if np.var(morph_data[iv, int_range]) > 0.02:
            mat_artifacts.append(iv)

    for iv in range(no_obj):
        outlier = sorted(outlier_tps[iv])
        good = sorted(good_tps[iv])
        int_range = range(int(cell_data[0, iv]), int(cell_data[0, iv]))
        while outlier:
            its = min(outlier)
            gtp = max([g for g in good if g < its], default=min([g for g in good if g > its], default=its))
            A = art_masks[its,:,:]
            
# =============================================================================
#             plt.figure()
#             plt.imshow(np.uint16(M), cmap='gray')
#             plt.title('M')
#             plt.show()
# =============================================================================
            
            M1 = MTrack[gtp] == (iv + 1)
            M2 = thin(M1, 30)
            M3 = A * M2
            
            plt.figure()
            plt.imshow(np.uint16(M3), cmap='gray')
            plt.title('M3')
            plt.show()
            
            indx = np.unique(A[M3 != 0])
            if indx.size > 0:
                X1 = np.zeros_like(MTrack[its,:,:])
                for itt2 in indx:
                    if np.sum(M3 == itt2) > 5:
                        X1[A == itt2] = 1
                X1 = binary_fill_holes(X1)
                X2 = label(X1)
                if np.max(X2) <= 1 and abs(np.sum(X1) - cor_data[0, iv]) <= 2 * cor_data[1, iv]:
                    MTrack[its][MTrack[its] == (iv + 1)] = 0
                    MTrack[its][X1 == 1] = iv + 1
                else:
                    MTrack[its][MTrack[its] == (iv + 1)] = 0
                    MTrack[its][MTrack[gtp] == (iv + 1)] = iv + 1
            outlier = [o for o in outlier if o != its]
            good.append(its)

    for iv in range(no_obj):
        if cell_data[1, iv] != tp_end:
            count = 0
            for its in range(int(cell_data[1, iv]), tp_end):
                A = art_masks[its,:,:]
                M1 = MTrack[its - 1] == (iv + 1)
                M2 = thin(M1, 30)
                M3 = A * M2
                indx = np.unique(A[M3 != 0])
                if indx.size > 0:
                    X1 = np.zeros_like(MTrack[its,:,:])
                    for itt2 in indx:
                        if np.sum(M3 == itt2) > 5:
                            X1[A == itt2] = 1
                    if abs(np.sum(X1) - cor_data[0, iv]) > 2 * cor_data[1, iv]:
                        count += 1
                        MTrack[its,:,:][MTrack[its - 1,:,:] == (iv + 1)] = iv + 1
                    else:
                        MTrack[its,:,:][X1 == 1] = iv + 1
                else:
                    count += 1
                    MTrack[its,:,:][MTrack[its - 1,:,:] == (iv + 1)] = iv + 1
            if count / (tp_end - cell_data[iv, 0]) > 0.8:
                mat_artifacts.append(iv + 1)

    # Remove cell artifacts and rename
    if mat_artifacts:
        all_ccel = list(range(1, no_obj + 1))
        mat_artifacts = sorted(set(mat_artifacts))
        for iv in mat_artifacts:
            for its in range(len(MTrack)):
                MTrack[its,:,:][MTrack[its,:,:] == iv] = 0
        good_cells = sorted(set(all_ccel) - set(mat_artifacts))
        for iv in range(len(good_cells)):
            for its in range(len(MTrack)):
                MTrack[its,:,:][MTrack[its,:,:] == good_cells[iv]] = iv + 1
        no_obj = len(good_cells)
        
        

    # Recalculating MAT Data
    all_obj_new = cal_allob1(no_obj, MTrack, list(range(len(MTrack))))
    cell_data_new = cal_celldata(all_obj_new, no_obj)

    cell_data = cell_data_new
    all_obj = all_obj_new
    Matmasks = MTrack

    sio.savemat(f'{sav_path}{pos}_MAT_16_18_Track1.mat', {
        "Matmasks": Matmasks,
        "all_obj": all_obj,
        "cell_data": cell_data,
        "no_obj": no_obj,
        "shock_period": shock_period,
        "mat_artifacts": mat_artifacts
    }, do_compression=True)
