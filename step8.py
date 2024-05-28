#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 00:19:31 2024

@author: samarth
"""

import os
import numpy as np
import scipy.io as sio
import functions.SR_240222_cal_allob as SR_240222_cal_allob

# Initialize variables
pos = 'Pos0_2'
path = os.path.join('E:', 'SR_Tracking', 'toy_data', pos, '')
sav_path = os.path.join('E:', 'SR_Tracking', 'toy_data', 'Tracks', '')

ARTfiles = [f for f in os.listdir(sav_path) if f.endswith('_ART_Track.mat')]
ART1files = [f for f in os.listdir(sav_path) if f.endswith('_ART_Track1.mat')]
MATfiles = [f for f in os.listdir(sav_path) if f.endswith('_MAT_16_18_Track1.mat')]
TETfiles = [f for f in os.listdir(sav_path) if f.endswith('_TET_Track.mat')]

st_inter = 150
end_inter = 183
last_tp = 282
last_tp_int = 777

# Function to process ART Tracks
def process_art_tracks(files, last_tp, end_inter, st_inter, suffix):
    for file in files:
        art_path = os.path.join(sav_path, file)
        data = sio.loadmat(art_path)
        Mask3 = data['Mask3']
        no_obj = data['no_obj'][0, 0]
        
        endi = Mask3.shape[1] - (last_tp - end_inter)
        tr1 = Mask3[:, :st_inter-1]
        tr2 = Mask3[:, st_inter:endi-1]
        tr3 = Mask3[:, endi:]
        
        ground_truth = []
        for i in range(tr2.shape[1] - 1):
            if i % 16 == 0:
                ground_truth.append(tr2[:, i])
        
        tr_final = np.hstack((tr1, np.array(ground_truth).T, tr3))
        Mask3 = tr_final
        
        new_art_name = file.replace('.mat', f'_{suffix}.mat')
        
        all_ob = SR_240222_cal_allob(no_obj, Mask3, range(Mask3.shape[1]))
        
        cell_artifacts = []
        cell_exists = np.zeros((2, all_ob.shape[0]))
        
        for itt2 in range(all_ob.shape[0]):
            if np.all(all_ob[itt2, :] == 0):
                cell_artifacts.append(itt2)
            else:
                cell_exists[0, itt2] = np.argmax(all_ob[itt2, :] > 0)
                cell_exists[1, itt2] = len(all_ob[itt2, :]) - 1 - np.argmax(all_ob[itt2, ::-1] > 0)
        
        if cell_artifacts:
            all_ccel = np.arange(1, no_obj + 1)
            good_cells = np.setdiff1d(all_ccel, cell_artifacts)
            good_cells = np.sort(good_cells)
            
            for iv in range(good_cells.size):
                for its in range(Mask3.shape[1]):
                    Mask3[:, its][Mask3[:, its] == good_cells[iv]] = iv + 1
            
            no_obj = good_cells.size
            all_ob = SR_240222_cal_allob(no_obj, Mask3, range(Mask3.shape[1]))
            cell_exists = np.zeros((2, all_ob.shape[0]))
            for itt2 in range(all_ob.shape[0]):
                cell_exists[0, itt2] = np.argmax(all_ob[itt2, :] > 0)
                cell_exists[1, itt2] = len(all_ob[itt2, :]) - 1 - np.argmax(all_ob[itt2, ::-1] > 0)
        
        sio.savemat(os.path.join(sav_path, new_art_name), {
            'Mask3': Mask3, 'all_ob': all_ob, 'ccell2': data['ccell2'], 
            'cell_exists': cell_exists, 'no_obj': no_obj
        }, do_compression=True)

# Process ART Tracks
process_art_tracks(ARTfiles, last_tp, end_inter, st_inter, 'DS')
process_art_tracks(ART1files, last_tp, end_inter, st_inter, 'DS')

# Function to process MAT Tracks
def process_mat_tracks(files, last_tp, end_inter, st_inter):
    for file in files:
        mat_path = os.path.join(sav_path, file)
        data = sio.loadmat(mat_path)
        Matmasks = data['Matmasks']
        no_obj = data['no_obj'][0, 0]
        shock_period = data['shock_period']
        
        endi = len(Matmasks) - (last_tp - end_inter)
        tr1 = Matmasks[:st_inter-1]
        tr2 = Matmasks[st_inter:endi-1]
        tr3 = Matmasks[endi:]
        
        ground_truth = []
        for i in range(len(tr2) - 1):
            if i % 16 == 0:
                ground_truth.append(tr2[i])
        
        tr_final = tr1 + ground_truth + tr3
        Matmasks = tr_final
        
        new_mat_name = file.replace('.mat', '_DS.mat')
        
        all_obj = SR_240222_cal_allob(no_obj, Matmasks, range(len(Matmasks)))
        
        cell_artifacts = []
        cell_data = np.zeros((all_obj.shape[0], 2))
        
        for itt2 in range(all_obj.shape[0]):
            if np.all(all_obj[itt2, :] == 0):
                cell_artifacts.append(itt2)
            else:
                cell_data[itt2, 0] = np.argmax(all_obj[itt2, :] > 0)
                cell_data[itt2, 1] = len(all_obj[itt2, :]) - 1 - np.argmax(all_obj[itt2, ::-1] > 0)
        
        if cell_artifacts:
            all_ccel = np.arange(1, no_obj + 1)
            good_cells = np.setdiff1d(all_ccel, cell_artifacts)
            good_cells = np.sort(good_cells)
            
            for iv in range(good_cells.size):
                for its in range(len(Matmasks)):
                    pix = np.where(Matmasks[its] == good_cells[iv])
                    Matmasks[its][pix] = iv + 1
            
            no_obj = good_cells.size
            all_obj = SR_240222_cal_allob(no_obj, Matmasks, range(len(Matmasks)))
            cell_data = np.zeros((all_obj.shape[0], 2))
            for itt2 in range(all_obj.shape[0]):
                cell_data[itt2, 0] = np.argmax(all_obj[itt2, :] > 0)
                cell_data[itt2, 1] = len(all_obj[itt2, :]) - 1 - np.argmax(all_obj[itt2, ::-1] > 0)
        
        sio.savemat(os.path.join(sav_path, new_mat_name), {
            'Matmasks': Matmasks, 'cell_data': cell_data, 'no_obj': no_obj, 
            'shock_period': shock_period, 'all_obj': all_obj
        }, do_compression=True)

# Process MAT Tracks
process_mat_tracks(MATfiles, last_tp, end_inter, st_inter)

# Function to process TET Tracks
def process_tet_tracks(files, last_tp, end_inter, st_inter):
    for file in files:
        tet_path = os.path.join(sav_path, file)
        data = sio.loadmat(tet_path)
        TETmasks = data['TETmasks']
        TET_obj = data['TET_obj'][0, 0]
        shock_period = data['shock_period']
        
        TETC = TETmasks
        for itx1 in range(last_tp_int):
            if itx1 >= len(TETmasks):
                TETC.append([])
        TETmasks = TETC
        
        endi = len(TETmasks) - (last_tp - end_inter)
        tr1 = TETmasks[:st_inter-1]
        tr2 = TETmasks[st_inter:endi-1]
        tr3 = TETmasks[endi:]
        
        ground_truth = []
        for i in range(len(tr2) - 1):
            if i % 16 == 0:
                ground_truth.append(tr2[i])
        
        tr_final = tr1 + ground_truth + tr3
        TETmasks = tr_final
        
        new_tet_name = file.replace('.mat', '_DS.mat')
        
        TET_Size = SR_240222_cal_allob(TET_obj, TETmasks, range(len(TETmasks)))
        
        cell_artifacts = []
        TET_exists = np.zeros((TET_Size.shape[0], 2))
        
        for itt2 in range(TET_Size.shape[0]):
            if np.all(TET_Size[itt2, :] == 0):
                cell_artifacts.append(itt2)
            else:
                TET_exists[itt2, 0] = np.argmax(TET_Size[itt2, :] > 0)
                TET_exists[itt2, 1] = len(TET_Size[itt2, :]) - 1 - np.argmax(TET_Size[itt2, ::-1] > 0)
        
        if cell_artifacts:
            all_ccel = np.arange(1, TET_obj + 1)
            good_cells = np.setdiff1d(all_ccel, cell_artifacts)
            good_cells = np.sort(good_cells)
            
            for iv in range(good_cells.size):
                for its in range(len(TETmasks)):
                    pix = np.where(TETmasks[its] == good_cells[iv])
                    TETmasks[its][pix] = iv + 1
            
            TET_obj = good_cells.size
            TET_Size = SR_240222_cal_allob(TET_obj, TETmasks, range(len(TETmasks)))
            TET_exists = np.zeros((TET_Size.shape[0], 2))
            for itt2 in range(TET_Size.shape[0]):
                TET_exists[itt2, 0] = np.argmax(TET_Size[itt2, :] > 0)
                TET_exists[itt2, 1] = len(TET_Size[itt2, :]) - 1 - np.argmax(TET_Size[itt2, ::-1] > 0)
        
        sio.savemat(os.path.join(sav_path, new_tet_name), {
            'TETmasks': TETmasks, 'shock_period': shock_period, 'TET_exists': TET_exists,
            'tet_masks_exists_tp': data['tet_masks_exists_tp'], 'TET_obj': TET_obj, 
            'TET_Size': TET_Size, 'thresh': data['thresh'], 'thresh_next_cell': data['thresh_next_cell'], 
            'thresh_perecent': data['thresh_perecent']
        }, do_compression=True)

# Process TET Tracks
process_tet_tracks(TETfiles, last_tp, end_inter, st_inter)
