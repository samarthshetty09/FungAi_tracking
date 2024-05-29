import os
import numpy as np
import scipy.io as sio
from skimage.morphology import binary_erosion, binary_thick, label
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from scipy.ndimage import distance_transform_edt as bwdist
from scipy.ndimage import watershed_ift as watershed
import cv2  # OpenCV for image processing

# Initialize variables
pos = 'Pos0_2'
path = os.path.join('E:', 'SR_Tracking', 'toy_data', pos, '')
sav_path = os.path.join('E:', 'SR_Tracking', 'toy_data', 'Tracks', '')

# Load TET Tracks
tet_track_path = os.path.join(sav_path, f'{pos}_TET_Track_DS')
file_list = [f for f in os.listdir(tet_track_path) if os.path.isfile(os.path.join(tet_track_path, f))]
tet = sio.loadmat(os.path.join(sav_path, file_list[0]))

tet_track_path = os.path.join(sav_path, f'{pos}_TET_ID')
file_list = [f for f in os.listdir(tet_track_path) if os.path.isfile(os.path.join(tet_track_path, f))]
tet_ids = sio.loadmat(os.path.join(sav_path, file_list[0]))

# Load ART Tracks
art_track_path = os.path.join(sav_path, f'{pos}_ART_Track_DS')
file_list = [f for f in os.listdir(art_track_path) if os.path.isfile(os.path.join(art_track_path, f))]
art = sio.loadmat(os.path.join(sav_path, file_list[0]))

masks = tet['TETmasks']
ART = art['Mask3']
shock_period = art['shock_period']
for i in range(len(masks)):
    if masks[i].size > 0:
        masks[i] = resize(masks[i], ART[i].shape, order=0, preserve_range=True).astype(int)

start = shock_period[0, 1] + 1
tp_end = len(ART)
int_range = range(start, tp_end)

# To determine if the TET cells are dead or alive after shock period
size_var_tet = np.zeros(tet['TET_obj'][0, 0])
dead_tets = np.zeros(tet['TET_obj'][0, 0])
for iv in range(tet['TET_obj'][0, 0]):
    if tet_ids['TET_ID'][0, iv] != -1:
        A = art['all_ob'][tet_ids['TET_ID'][0, iv], shock_period[0, 1] + 1:].flatten()
        LT, ST, R = trenddecomp(A)
        size_var_tet[iv] = np.var(ST)
        dead_tets[iv] = 1 if np.var(ST) < 1000 else 0
    else:
        size_var_tet[iv] = -10**5
        dead_tets[iv] = -10**5

# Finding the first time when TETs germinate and new cells begin to show up
begin = 0
for its in int_range:
    if its == max(int_range):
        begin = 0
        break
    A1 = ART[its]
    A2 = ART[its + 1]
    A3 = (A1 > 0).astype(int) * A2.astype(int)
    indx_ori = np.unique(A1[A1 != 0])
    indx_new = np.unique(A2[A2 != 0])
    vals = np.setdiff1d(indx_new, indx_ori)
    if vals.size > 0:
        begin = its + 1
        break

# Dividing the FOV into regions that belong to certain tetrads using the watershed algorithm
if begin != 0:
    int1 = range(begin, len(ART))
    new_indx = [set() for _ in range(len(ART))]
    
    for its in int1:
        A1 = ART[its - 1]
        A2 = ART[its]
        indx_ori = np.unique(A1[A1 != 0])
        indx_new = np.unique(A2[A2 != 0])
        vals = np.setdiff1d(indx_new, indx_ori)
        new_indx[its] = set(vals)
    
    kka = 0
    new_indx_new = []
    for vals in new_indx:
        if vals:
            kka += 1
            new_indx_new.append(vals)
    
    new_born = np.unique(np.hstack(new_indx_new))
    
    I2 = np.zeros(ART[start].shape, dtype=int)
    for ccell in range(tet['TET_obj'][0, 0]):
        if tet_ids['TET_ID'][0, ccell] != -1:
            if dead_tets[ccell] == 0:
                if tet['TET_exists'][ccell, 1] >= shock_period[0, 1] + 1:
                    stats = regionprops(masks[shock_period[0, 1] + 1] == ccell + 1)
                else:
                    stats = regionprops(masks[tet['TET_exists'][ccell, 1]] == ccell + 1)
                if stats:
                    cent = np.round(stats[0].centroid).astype(int)
                    I2[cent[1], cent[0]] = 1
    
    I21 = binary_thick(I2, structure=np.ones((9, 9)))
    I4 = bwdist(I21)
    I3 = watershed(I4.astype(np.uint8), markers=4)

# Checking which cell from ART tracks belongs to which region created using watershed
region = []
amt = []
k = 0
for iv in range(art['no_obj'][0, 0]):
    I12 = np.zeros(ART[start].shape, dtype=int)
    kx = 0
    for its in int1:
        I11 = (ART[its] == iv + 1).astype(int)
        if I11.sum() > 0:
            kx += 1
            if 1 <= kx <= 2:
                I11 *= 1000
        I12 += I11
    I13 = (I12 > 0).astype(int) * I3.astype(int)
    pix = np.unique(I13[I13 != 0])
    if pix.size > 0:
        for p in pix:
            amt.append([iv, p, (I13 == p).sum()])
        k += 1
        region.append([iv, pix[np.argmax([a[2] for a in amt if a[0] == iv])]])

unique_regions = np.unique([r[1] for r in region])
cell_arrays = [np.array([r[0] for r in region if r[1] == ur], dtype=int) for ur in unique_regions]

# Saving the TET ID, TET regions, and possible descendants
descendants = [set(ci) for ci in cell_arrays]
for iv in range(tet['TET_obj'][0, 0]):
    if tet_ids['TET_ID'][0, iv] != -1:
        if dead_tets[iv] == 0:
            T1 = masks[tet['TET_exists'][iv, 0]] == iv + 1
            T2 = (I3 * T1.astype(int)).astype(int)
            pix = np.unique(T2[T2 != 0])
            if pix.size > 0:
                amt1 = [(iv, p, (T2 == p).sum()) for p in pix]
                tet_region = pix[np.argmax([a[2] for a in amt1])]
                common_indices = np.intersect1d(new_born, cell_arrays[tet_region - 1])
                common_indices = np.append(common_indices, tet_ids['TET_ID'][0, iv])
            else:
                common_indices = np.array([tet_ids['TET_ID'][0, iv]])
        else:
            common_indices = np.array([tet_ids['TET_ID'][0, iv]])
    else:
        common_indices = np.array([tet_ids['TET_ID'][0, iv]])

    descendants[iv] = set(common_indices)

alive_tets = [iv for iv in range(tet['TET_obj'][0, 0]) if dead_tets[iv] == 0]

# Identifying incorrectly associated descendants and reassigning
need_remov = []
works = []
for iv in alive_tets:
    common_indices1 = list(descendants[iv])
    for ittx1 in common_indices1:
        if ittx1 in tet_ids['TET_ID']:
            continue
        its = art['cell_exists'][0, ittx1 - 1]
        M = ART[its]
        I_s0 = np.zeros_like(M, dtype=int)
        I_s2 = np.zeros_like(M, dtype=int)
        for rem in need_remov:
            common_indices1 = list(set(common_indices1) - set(rem))
        for it in common_indices1:
            I_s2 = (M == it).astype(int)
            I_s2 = binary_thick(I_s2, structure=np.ones((5, 5)))
            I_s0 += I_s2
        IA1 = (I_s0 > 0).astype(int)
        IA2 = label(IA1)
        if IA2.max() > 1:
            out = np.unique(IA2)
            sizes_occup = [(itt, (IA2 == itt).sum()) for itt in out if itt != 0]
            xx1 = max(sizes_occup, key=lambda x: x[1])[0]
            AAB = 0
            for itt2 in out:
                if itt2 != 0:
                    AB1 = IA2.copy()
                    AB2 = binary_thick(M == ittx1, structure=np.ones((5, 5)))
                    AB3 = AB1 * AB2
                    pixab = np.unique(AB3)
                    if pixab.size > 1 and pixab[1] == xx1:
                        continue
                    elif pixab.size == 1:
                        continue
                    else:
                        AAB = 1
            if AAB == 1:
                for itx in alive_tets:
                    if itx != iv:
                        M = ART[its]
                        I_s0 = np.zeros_like(M, dtype=int)
                        I_s2 = np.zeros_like(M, dtype=int)
                        for it in common_indices1 + [ittx1]:
                            I_s2 = (M == it).astype(int)
                            I_s2 = binary_thick(I_s2, structure=np.ones((3, 3)))
                            I_s0 += I_s2
                        IA11 = (I_s0 > 0).astype(int)
                        IA21 = label(IA11)
                        if IA21.max() == 1:
                            works.append([ittx1, iv, itx])
                        else:
                            need_remov.append([ittx1, iv])

# Updating descendants based on reassignment
for work in works:
    ittx1, old_iv, new_iv = work
    descendants[old_iv].remove(ittx1)
    descendants[new_iv].add(ittx1)
for rem in need_remov:
    ittx1, old_iv = rem
    descendants[old_iv].remove(ittx1)

descendants_data = []
for iv in range(tet['TET_obj'][0, 0]):
    if tet_ids['TET_ID'][0, iv] == -1:
        descendants_data.append([iv, tet_ids['TET_ID'][0, iv], -1])
    else:
        descendants_data.append([iv, tet_ids['TET_ID'][0, iv], list(descendants[iv] - {tet_ids['TET_ID'][0, iv]})])

sio.savemat(os.path.join(sav_path, f'{pos}_descendants_new_art.mat'), {
    "I3": I3, "descendants_data": descendants_data, "descendants": descendants, 
    "alive_tets": alive_tets, "common_indices": common_indices, "cell_arrays": cell_arrays, 
    "TET_obj": tet['TET_obj']
}, do_compression=True)
