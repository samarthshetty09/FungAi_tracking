#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:07:22 2024

@author: samarth
"""

import os
import glob
import numpy as np
from scipy.io import loadmat, savemat
from skimage.io import imread
from skimage.filters import median
from skimage.morphology import remove_small_objects, binary_opening
from skimage.measure import label
from skimage.morphology import binary_closing
from skimage.morphology import binary_fill_holes
from skimage.util import img_as_float
from scipy.ndimage import median_filter
import multiprocessing as mp
from functools import partial

def get_wind_coord1(ccell, cell_margin):
    # TODO!
    pass

def OAM_230905_Get_Sphere_Vol_cell(ccell):
    # TODO!
    pass

def OAM_230906_Gaussian_nuclear_fit(I_cell, peak_cutoff, x_size, y_size, ccell):
    # TODO!
    pass

def OAM_230905_Get_Sphere_Vol_nuc(mask_nuc):
    # TODO!
    pass

def OAM_230905_Get_Sphere_Vol_cyt(mask_cyt):
    # TODO!
    pass

def process_image(exp_fold_name, path_h0, im_path, I_mean_modifier, peak_cutoff, cell_margin):
    # load tracks
    mat_data = loadmat(os.path.join(path_h0, f"{exp_fold_name}_ART_Track0_122.mat"))
    Mask2 = mat_data.get('Mask7')  # or Mask6 if needed

    x_size, y_size, _ = Mask2.shape
    Ipath = os.path.join(im_path, exp_fold_name)
    file_n = glob.glob(os.path.join(Ipath, '*.tif'))
    file_n2 = [os.path.basename(f) for f in file_n]

    # Extract channel names
    Name = []
    for it01 in range(min(7, len(file_n2))):
        if 'Ph3' not in file_n2[it01]:
            channelName = file_n2[it01][14:]
            Name.append(channelName)

    channels = list(set(Name))
    
    # Allocate the variable names
    ALLDATA = {
        1: ['Channel_name'],
        2: ['Cell_Size'],
        3: ['cell_vol'],
        4: ['max_nuc_int1'],
        5: ['mean_cell_Fl1'],
        6: ['Conc_T_cell_Fl1'],
        7: ['mem_area1'],
        8: ['nuc_area1'],
        9: ['cyt_area1'],
        10: ['mean_fl_mem1'],
        11: ['std_fl_mem1'],
        12: ['tot_Fl_mem1'],
        13: ['tot_Fl_cyt1'],
        14: ['tot_Fl_nuc1'],
        15: ['mean_int_per_area_C1'],
        16: ['mean_int_per_area_N1'],
        17: ['nuc_Vol1'],
        18: ['cyt_Vol1'],
        19: ['cyt_Vol_sub1'],
        20: ['FL_Conc_T1'],
        21: ['FL_Conc_C1'],
        22: ['FL_Conc_N1'],
        23: ['FL_mean_int_N_thr1'],
        24: ['FL_mean_int_C_thr1']
    }

    for it1, channel in enumerate(channels):
        file1 = glob.glob(os.path.join(Ipath, f"*{channel}"))
        file2 = [os.path.basename(f) for f in file1]

        no_obj = np.unique(Mask2).size
        cell_Vol1 = np.zeros((no_obj, Mask2.shape[2]))
        max_nuc_int1 = np.zeros((no_obj, Mask2.shape[2]))
        mean_cell_Fl1 = np.zeros((no_obj, Mask2.shape[2]))
        Conc_T_cell_Fl1 = np.zeros((no_obj, Mask2.shape[2]))
        mem_area1 = np.zeros((no_obj, Mask2.shape[2]))
        nuc_area1 = np.zeros((no_obj, Mask2.shape[2]))
        cyt_area1 = np.zeros((no_obj, Mask2.shape[2]))
        mean_fl_mem1 = np.zeros((no_obj, Mask2.shape[2]))
        std_fl_mem1 = np.zeros((no_obj, Mask2.shape[2]))
        tot_Fl_mem1 = np.zeros((no_obj, Mask2.shape[2]))
        tot_Fl_cyt1 = np.zeros((no_obj, Mask2.shape[2]))
        tot_Fl_nuc1 = np.zeros((no_obj, Mask2.shape[2]))
        mean_int_per_area_C1 = np.zeros((no_obj, Mask2.shape[2]))
        mean_int_per_area_N1 = np.zeros((no_obj, Mask2.shape[2]))
        nuc_Vol1 = np.zeros((no_obj, Mask2.shape[2]))
        cyt_Vol1 = np.zeros((no_obj, Mask2.shape[2]))
        cyt_Vol_sub1 = np.zeros((no_obj, Mask2.shape[2]))
        FL_Conc_T1 = np.zeros((no_obj, Mask2.shape[2]))
        FL_Conc_C1 = np.zeros((no_obj, Mask2.shape[2]))
        FL_Conc_N1 = np.zeros((no_obj, Mask2.shape[2]))
        FL_mean_int_N_thr1 = np.zeros((no_obj, Mask2.shape[2]))
        FL_mean_int_C_thr1 = np.zeros((no_obj, Mask2.shape[2]))
        all_back = np.zeros((1, Mask2.shape[2]))
        Cell_Size1 = np.zeros((no_obj, Mask2.shape[2]))

        def process_cell(c_time, file2, Ipath, Mask2, I_mean_modifier, peak_cutoff, cell_margin, x_size, y_size):
            results = {}

            Lcells = Mask2[:, :, c_time]

            if Lcells is not None and np.sum(Lcells) != 0:
                I = imread(os.path.join(Ipath, file2[c_time]))
                I = img_as_float(I)
                I = median_filter(I, size=(3, 3))
                
                # background correction
                bck = I * (~Lcells.astype(bool))
                backgr = np.median(bck[bck != 0])
                I -= backgr
                all_back[0, c_time] = backgr
                
                # Allocation for parfor loop sliced variables
                cell_Vol = np.zeros(no_obj)
                max_nuc_int = np.zeros(no_obj)
                mean_cell_Fl = np.zeros(no_obj)
                Conc_T_cell_Fl = np.zeros(no_obj)
                mem_area = np.zeros(no_obj)
                nuc_area = np.zeros(no_obj)
                cyt_area = np.zeros(no_obj)
                mean_fl_mem = np.zeros(no_obj)
                std_fl_mem = np.zeros(no_obj)
                tot_Fl_mem = np.zeros(no_obj)
                tot_Fl_cyt = np.zeros(no_obj)
                tot_Fl_nuc = np.zeros(no_obj)
                mean_int_per_area_C = np.zeros(no_obj)
                mean_int_per_area_N = np.zeros(no_obj)
                nuc_Vol = np.zeros(no_obj)
                cyt_Vol = np.zeros(no_obj)
                cyt_Vol_sub = np.zeros(no_obj)
                FL_Conc_T = np.zeros(no_obj)
                FL_Conc_C = np.zeros(no_obj)
                FL_Conc_N = np.zeros(no_obj)
                FL_mean_int_N_thr = np.zeros(no_obj)
                FL_mean_int_C_thr = np.zeros(no_obj)
                cell_size = np.zeros(no_obj)

                for cell_no in range(no_obj):
                    ccell = (Lcells == cell_no).astype(np.float64)

                    if np.sum(ccell) != 0:
                        x_cn, y_cn = get_wind_coord1(ccell, cell_margin)
                        ccell = ccell[y_cn, x_cn]
                        cell_size[cell_no] = np.sum(ccell)
                        cell_Vol[cell_no] = OAM_230905_Get_Sphere_Vol_cell(ccell)

                        I_cell = I[y_cn, x_cn]
                        put_I = ccell * I_cell
                        max_nuc_int[cell_no] = np.max(put_I)
                        mean_cell_Fl[cell_no] = np.sum(put_I) / np.sum(ccell)
                        Conc_T_cell_Fl[cell_no] = np.sum(put_I) / cell_Vol[cell_no]

                        mask_nuc = OAM_230906_Gaussian_nuclear_fit(I_cell, peak_cutoff, x_size, y_size, ccell)

                        mask_mem = binary_opening(ccell, np.ones((3, 3)))
                        mem_area[cell_no] = np.sum(mask_mem)

                        if np.sum(mask_nuc) != 0:
                            mask_cyt = ccell - mask_nuc
                        else:
                            mask_cyt = np.nan

                        nuc_area[cell_no] = np.sum(mask_nuc)
                        cyt_area[cell_no] = np.sum(mask_cyt)
                        mem_fl = mask_mem * I_cell
                        mean_fl_mem[cell_no] = np.median(mem_fl[mem_fl != 0])
                        std_fl_mem[cell_no] = np.std(mem_fl[mem_fl != 0])
                        tot_Fl_mem[cell_no] = np.sum(mem_fl)
                        tot_Fl_cyt[cell_no] = np.sum(mask_cyt * I_cell)
                        tot_Fl_nuc[cell_no] = np.sum(mask_nuc * I_cell)
                        mean_int_per_area_C[cell_no] = np.sum(mask_cyt * I_cell) / np.sum(mask_cyt)
                        mean_int_per_area_N[cell_no] = np.sum(mask_nuc * I_cell) / nuc_area[cell_no]
                        nuc_Vol[cell_no] = OAM_230905_Get_Sphere_Vol_nuc(mask_nuc)
                        cyt_Vol[cell_no] = OAM_230905_Get_Sphere_Vol_cyt(mask_cyt)
                        cyt_Vol_sub[cell_no] = cell_Vol[cell_no] - nuc_Vol[cell_no]
                        FL_Conc_T[cell_no] = np.sum(put_I) / cell_Vol[cell_no]
                        FL_Conc_C[cell_no] = tot_Fl_cyt[cell_no] / cyt_Vol[cell_no]
                        FL_Conc_N[cell_no] = tot_Fl_nuc[cell_no] / nuc_Vol[cell_no]
                        
                        
                        # obtain nucleus by threshold constructed with "I_mean_modifier"
                        put_mod = (put_I > (I_mean_modifier * np.mean(put_I[put_I > 0])))
                        put_mod = remove_small_objects(put_mod, min_size=5, connectivity=2)
                        put_mod = binary_fill_holes(put_mod)
                        FL_mean_int_N_thr[cell_no] = np.sum(put_mod * put_I) / np.sum(put_mod)
                        no = put_I * (~put_mod)
                        FL_mean_int_C_thr[cell_no] = np.sum(no[no > 0]) / np.sum(no[no > 0] > 0)

                results['cell_Vol'] = cell_Vol
                results['max_nuc_int'] = max_nuc_int
                results['mean_cell_Fl'] = mean_cell_Fl
                results['Conc_T_cell_Fl'] = Conc_T_cell_Fl
                results['mem_area'] = mem_area
                results['nuc_area'] = nuc_area
                results['cyt_area'] = cyt_area
                results['mean_fl_mem'] = mean_fl_mem
                results['std_fl_mem'] = std_fl_mem
                results['tot_Fl_mem'] = tot_Fl_mem
                results['tot_Fl_cyt'] = tot_Fl_cyt
                results['tot_Fl_nuc'] = tot_Fl_nuc
                results['mean_int_per_area_C'] = mean_int_per_area_C
                results['mean_int_per_area_N'] = mean_int_per_area_N
                results['nuc_Vol'] = nuc_Vol
                results['cyt_Vol'] = cyt_Vol
                results['cyt_Vol_sub'] = cyt_Vol_sub
                results['FL_Conc_T'] = FL_Conc_T
                results['FL_Conc_C'] = FL_Conc_C
                results['FL_Conc_N'] = FL_Conc_N
                results['FL_mean_int_N_thr'] = FL_mean_int_N_thr
                results['FL_mean_int_C_thr'] = FL_mean_int_C_thr
                results['cell_size'] = cell_size

            else:
                results = {
                    'cell_Vol': np.zeros(no_obj),
                    'max_nuc_int': np.zeros(no_obj),
                    'mean_cell_Fl': np.zeros(no_obj),
                    'Conc_T_cell_Fl': np.zeros(no_obj),
                    'mem_area': np.zeros(no_obj),
                    'nuc_area': np.zeros(no_obj),
                    'cyt_area': np.zeros(no_obj),
                    'mean_fl_mem': np.zeros(no_obj),
                    'std_fl_mem': np.zeros(no_obj),
                    'tot_Fl_mem': np.zeros(no_obj),
                    'tot_Fl_cyt': np.zeros(no_obj),
                    'tot_Fl_nuc': np.zeros(no_obj),
                    'mean_int_per_area_C': np.zeros(no_obj),
                    'mean_int_per_area_N': np.zeros(no_obj),
                    'nuc_Vol': np.zeros(no_obj),
                    'cyt_Vol': np.zeros(no_obj),
                    'cyt_Vol_sub': np.zeros(no_obj),
                    'FL_Conc_T': np.zeros(no_obj),
                    'FL_Conc_C': np.zeros(no_obj),
                    'FL_Conc_N': np.zeros(no_obj),
                    'FL_mean_int_N_thr': np.zeros(no_obj),
                    'FL_mean_int_C_thr': np.zeros(no_obj),
                    'cell_size': np.zeros(no_obj)
                }

            return results

        with mp.Pool(processes=16) as pool:
            results = pool.map(partial(process_cell, file2=file2, Ipath=Ipath, Mask2=Mask2, I_mean_modifier=I_mean_modifier,
                                       peak_cutoff=peak_cutoff, cell_margin=cell_margin, x_size=x_size, y_size=y_size), range(Mask2.shape[2]))

        for c_time, result in enumerate(results):
            cell_Vol1[:, c_time] = result['cell_Vol']
            max_nuc_int1[:, c_time] = result['max_nuc_int']
            mean_cell_Fl1[:, c_time] = result['mean_cell_Fl']
            Conc_T_cell_Fl1[:, c_time] = result['Conc_T_cell_Fl']
            mem_area1[:, c_time] = result['mem_area']
            nuc_area1[:, c_time] = result['nuc_area']
            cyt_area1[:, c_time] = result['cyt_area']
            mean_fl_mem1[:, c_time] = result['mean_fl_mem']
            std_fl_mem1[:, c_time] = result['std_fl_mem']
            tot_Fl_mem1[:, c_time] = result['tot_Fl_mem']
            tot_Fl_cyt1[:, c_time] = result['tot_Fl_cyt']
            tot_Fl_nuc1[:, c_time] = result['tot_Fl_nuc']
            mean_int_per_area_C1[:, c_time] = result['mean_int_per_area_C']
            mean_int_per_area_N1[:, c_time] = result['mean_int_per_area_N']
            nuc_Vol1[:, c_time] = result['nuc_Vol']
            cyt_Vol1[:, c_time] = result['cyt_Vol']
            cyt_Vol_sub1[:, c_time] = result['cyt_Vol_sub']
            FL_Conc_T1[:, c_time] = result['FL_Conc_T']
            FL_Conc_C1[:, c_time] = result['FL_Conc_C']
            FL_Conc_N1[:, c_time] = result['FL_Conc_N']
            FL_mean_int_N_thr1[:, c_time] = result['FL_mean_int_N_thr']
            FL_mean_int_C_thr1[:, c_time] = result['FL_mean_int_C_thr']
            Cell_Size1[:, c_time] = result['cell_size']

        ALLDATA[1].append(channel)
        ALLDATA[2].append(Cell_Size1)
        ALLDATA[3].append(cell_Vol1)
        ALLDATA[4].append(max_nuc_int1)
        ALLDATA[5].append(mean_cell_Fl1)
        ALLDATA[6].append(Conc_T_cell_Fl1)
        ALLDATA[7].append(mem_area1)
        ALLDATA[8].append(nuc_area1)
        ALLDATA[9].append(cyt_area1)
        ALLDATA[10].append(mean_fl_mem1)
        ALLDATA[11].append(std_fl_mem1)
        ALLDATA[12].append(tot_Fl_mem1)
        ALLDATA[13].append(tot_Fl_cyt1)
        ALLDATA[14].append(tot_Fl_nuc1)
        ALLDATA[15].append(mean_int_per_area_C1)
        ALLDATA[16].append(mean_int_per_area_N1)
        ALLDATA[17].append(nuc_Vol1)
        ALLDATA[18].append(cyt_Vol1)
        ALLDATA[19].append(cyt_Vol_sub1)
        ALLDATA[20].append(FL_Conc_T1)
        ALLDATA[21].append(FL_Conc_C1)
        ALLDATA[22].append(FL_Conc_N1)
        ALLDATA[23].append(FL_mean_int_N_thr1)
        ALLDATA[24].append(FL_mean_int_C_thr1)

    save_fold = os.path.join('FL_extracts', exp_name)
    os.makedirs(save_fold, exist_ok=True)
    path_save = os.path.join(path_h0, save_fold)
    name3 = f"{exp_fold_name}_FLEX.mat"
    savemat(os.path.join(path_save, name3), {'ALLDATA': ALLDATA, 'all_back': all_back, 'I_mean_modifier': I_mean_modifier,
                                             'peak_cutoff': peak_cutoff, 'cell_margin': cell_margin})

if __name__ == "__main__":
    exp_name = 'OAM_200303_6c_I'
    path_h0 = os.path.join('C:', 'Users', 'oargell', 'Videos', '6c', exp_name, 'Segs', 'ART', 'Tracks')
    im_path = os.path.join('C:', 'Users', 'oargell', 'Videos', '6c', exp_name)

    exp_foldrs = [d for d in os.listdir(im_path) if os.path.isdir(os.path.join(im_path, d))]
    exp_foldrs = [d for d in exp_foldrs if d not in ['.', '..', 'Tracks', 'Segs', 'X', 'CORRECT', 'Interpol']]
    exp_fold_name = exp_foldrs

    I_mean_modifier = 1.5
    peak_cutoff = 0.75
    cell_margin = 5

    with mp.Pool(processes=16) as pool:
        pool.starmap(process_image, [(name, path_h0, im_path, I_mean_modifier, peak_cutoff, cell_margin) for name in exp_fold_name])
