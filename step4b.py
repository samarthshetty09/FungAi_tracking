import os
import h5py
import numpy as np
import scipy.io as sio
from scipy.ndimage import median_filter, binary_fill_holes
from skimage.morphology import remove_small_objects, skeletonize
from skimage.io import imread
from concurrent.futures import ThreadPoolExecutor
import glob
from functions.OAM_230905_Get_Sphere_Vol_cyt import OAM_230905_Get_Sphere_Vol_cyt
from functions.OAM_230905_Get_Sphere_Vol_nuc import OAM_230905_Get_Sphere_Vol_nuc
from functions.OAM_230906_Gaussian_nuclear_fit import OAM_230906_Gaussian_nuclear_fit
from functions.get_wind_coord1 import get_wind_coord1

def OAM_230905_Get_Sphere_Vol_cell(ccell):
    # Dummy implementation
    return np.sum(ccell)

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

def resolve_h5py_reference(data, f):
    if isinstance(data, h5py.Reference):
        return f[data][()]
    return data

# Parameters
exp_name = 'OAM_200303_6c_I'
pos = 'Pos0_2'
# path_h0 = os.path.join('Users', 'samarth', 'Documents', 'MATLAB', exp_name, 'Segs', 'ART', 'Tracks')
path_h0 = '/Users/samarth/Documents/MATLAB/Full_Life_Cycle_tracking/Tracks2/'
im_path = f'/Users/samarth/Documents/MATLAB/Full_Life_Cycle_tracking/{pos}/'

# Get all directories
# exp_foldrs = [d for d in os.listdir(im_path) if os.path.isdir(os.path.join(im_path, d))]
# exclude_dirs = {'.', '..', 'Tracks', 'Segs', 'X', 'CORRECT', 'Interpol'}
# exp_foldrs = [d for d in exp_foldrs if d not in exclude_dirs and not d.endswith('.mat')]
# exp_fold_name = exp_foldrs

I_mean_modifier = 1.5
peak_cutoff = 0.75
cell_margin = 5

# Main processing loop
#for exp_folder in exp_fold_name:
    
    
mat_path = os.path.join(path_h0)
file_list = [f for f in os.listdir(mat_path) if '_ART_Track' in f]
sorted(file_list)
mat_data = load_mat(os.path.join(mat_path, file_list[0]))

  # change to Mask7 
Mask2 = []
with h5py.File(os.path.join(mat_path, file_list[0]), 'r') as f:
    for i in range(len(f['Mask3'])):
        masks_refs = f['Mask3'][i]
        for ref in masks_refs:
            mask = resolve_h5py_reference(ref, f)
            Mask2.append(mask)

# x_size, y_size = Mask2.shape[0], Mask2.shape[1]
x_size, y_size = Mask2[0].shape[0], Mask2[0].shape[1]



Ipath = os.path.join(im_path)

file_n = sorted(glob.glob(os.path.join(Ipath, '*.tif')))
no_obj = np.unique(Mask2).size

channels = [os.path.splitext(os.path.basename(f))[0][13:] for f in file_n if 'Ph3' not in f]
channels = np.unique(channels)

ALLDATA = [
    ['Channel_name'], ['Cell_Size'], ['cell_vol'], ['max_nuc_int1'], ['mean_cell_Fl1'],
    ['Conc_T_cell_Fl1'], ['mem_area1'], ['nuc_area1'], ['cyt_area1'], ['mean_fl_mem1'],
    ['std_fl_mem1'], ['tot_Fl_mem1'], ['tot_Fl_cyt1'], ['tot_Fl_nuc1'], ['mean_int_per_area_C1'],
    ['mean_int_per_area_N1'], ['nuc_Vol1'], ['cyt_Vol1'], ['cyt_Vol_sub1'], ['FL_Conc_T1'],
    ['FL_Conc_C1'], ['FL_Conc_N1'], ['FL_mean_int_N_thr1'], ['FL_mean_int_C_thr1']
]

for channel in channels:
    file1 = sorted(glob.glob(os.path.join(Ipath, f'*{channel}')))
    
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

    def process_timepoint(c_time):
        nonlocal cell_Vol1, max_nuc_int1, mean_cell_Fl1, Conc_T_cell_Fl1, mem_area1
        nonlocal nuc_area1, cyt_area1, mean_fl_mem1, std_fl_mem1, tot_Fl_mem1
        nonlocal tot_Fl_cyt1, tot_Fl_nuc1, mean_int_per_area_C1, mean_int_per_area_N1
        nonlocal nuc_Vol1, cyt_Vol1, cyt_Vol_sub1, FL_Conc_T1, FL_Conc_C1, FL_Conc_N1
        nonlocal FL_mean_int_N_thr1, FL_mean_int_C_thr1, Cell_Size1, all_back

        Lcells = Mask2[:, :, c_time]

        if Lcells.size != 0 and np.sum(Lcells) != 0:
            I = imread(file1[c_time])
            I = I.astype(np.float64)
            I = median_filter(I, size=3, mode='nearest')
            
            bck = I * (Lcells == 0)
            backgr = np.median(bck[bck != 0])
            I = I - backgr
            all_back[0, c_time] = backgr

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

            for cell_no in range(1, no_obj + 1):
                ccell = (Lcells == cell_no).astype(np.float64)
                if np.sum(ccell) != 0:
                    y_cn, x_cn = get_wind_coord1(ccell, cell_margin)
                    ccell = ccell[y_cn, x_cn]
                    cell_size[cell_no - 1] = np.sum(ccell)
                    cell_Vol[cell_no - 1] = OAM_230905_Get_Sphere_Vol_cell(ccell)

                    I_cell = I[y_cn, x_cn]
                    put_I = ccell * I_cell
                    max_nuc_int[cell_no - 1] = np.max(put_I)
                    mean_cell_Fl[cell_no - 1] = np.sum(put_I) / np.sum(ccell)
                    Conc_T_cell_Fl[cell_no - 1] = np.sum(put_I) / cell_Vol[cell_no - 1]

                    mask_nuc = OAM_230906_Gaussian_nuclear_fit(I_cell, peak_cutoff, x_size, y_size, ccell)
                    mask_mem = skeletonize(ccell).astype(np.float64)
                    mem_area[cell_no - 1] = np.sum(mask_mem)

                    if np.sum(mask_nuc) != 0:
                        mask_cyt = ccell - mask_nuc
                    else:
                        mask_cyt = np.nan

                    nuc_area[cell_no - 1] = np.sum(mask_nuc)
                    cyt_area[cell_no - 1] = np.sum(mask_cyt)
                    mem_fl = mask_mem * I_cell
                    mean_fl_mem[cell_no - 1] = np.median(mem_fl[mem_fl != 0])
                    std_fl_mem[cell_no - 1] = np.std(mem_fl[mem_fl != 0])
                    tot_Fl_mem[cell_no - 1] = np.sum(mem_fl)
                    tot_Fl_cyt[cell_no - 1] = np.sum(mask_cyt * I_cell)
                    tot_Fl_nuc[cell_no - 1] = np.sum(mask_nuc * I_cell)
                    mean_int_per_area_C[cell_no - 1] = np.sum(mask_cyt * I_cell) / np.sum(mask_cyt)
                    mean_int_per_area_N[cell_no - 1] = np.sum(mask_nuc * I_cell) / np.sum(mask_nuc)
                    nuc_Vol[cell_no - 1] = OAM_230905_Get_Sphere_Vol_nuc(mask_nuc)
                    cyt_Vol[cell_no - 1] = OAM_230905_Get_Sphere_Vol_cyt(mask_cyt)
                    cyt_Vol_sub[cell_no - 1] = cell_Vol[cell_no - 1] - nuc_Vol[cell_no - 1]
                    FL_Conc_T[cell_no - 1] = np.sum(put_I) / cell_Vol[cell_no - 1]
                    FL_Conc_C[cell_no - 1] = tot_Fl_cyt[cell_no - 1] / cyt_Vol[cell_no - 1]
                    FL_Conc_N[cell_no - 1] = tot_Fl_nuc[cell_no - 1] / nuc_Vol[cell_no - 1]

                    put_mod = (put_I > (I_mean_modifier * np.mean(put_I[put_I > 0]))).astype(np.float64)
                    put_mod = remove_small_objects(put_mod.astype(bool), 5).astype(np.float64)
                    put_mod = binary_fill_holes(put_mod)
                    FL_mean_int_N_thr[cell_no - 1] = np.sum(put_mod * put_I) / np.sum(put_mod)
                    no = put_I * (1 - put_mod)
                    FL_mean_int_C_thr[cell_no - 1] = np.sum(no[no > 0]) / np.sum(no > 0)
                else:
                    cell_Vol[cell_no - 1] = 0
                    max_nuc_int[cell_no - 1] = 0
                    mean_cell_Fl[cell_no - 1] = 0
                    Conc_T_cell_Fl[cell_no - 1] = 0
                    mem_area[cell_no - 1] = 0
                    nuc_area[cell_no - 1] = 0
                    cyt_area[cell_no - 1] = 0
                    mean_fl_mem[cell_no - 1] = 0
                    std_fl_mem[cell_no - 1] = 0
                    tot_Fl_mem[cell_no - 1] = 0
                    tot_Fl_cyt[cell_no - 1] = 0
                    tot_Fl_nuc[cell_no - 1] = 0
                    mean_int_per_area_C[cell_no - 1] = 0
                    mean_int_per_area_N[cell_no - 1] = 0
                    nuc_Vol[cell_no - 1] = 0
                    cyt_Vol[cell_no - 1] = 0
                    cyt_Vol_sub[cell_no - 1] = 0
                    FL_Conc_T[cell_no - 1] = 0
                    FL_Conc_C[cell_no - 1] = 0
                    FL_Conc_N[cell_no - 1] = 0
                    FL_mean_int_N_thr[cell_no - 1] = 0
                    FL_mean_int_C_thr[cell_no - 1] = 0
                    cell_size[cell_no - 1] = 0

            cell_Vol1[:, c_time] = cell_Vol
            max_nuc_int1[:, c_time] = max_nuc_int
            mean_cell_Fl1[:, c_time] = mean_cell_Fl
            Conc_T_cell_Fl1[:, c_time] = Conc_T_cell_Fl
            mem_area1[:, c_time] = mem_area
            nuc_area1[:, c_time] = nuc_area
            cyt_area1[:, c_time] = cyt_area
            mean_fl_mem1[:, c_time] = mean_fl_mem
            std_fl_mem1[:, c_time] = std_fl_mem
            tot_Fl_mem1[:, c_time] = tot_Fl_mem
            tot_Fl_cyt1[:, c_time] = tot_Fl_cyt
            tot_Fl_nuc1[:, c_time] = tot_Fl_nuc
            mean_int_per_area_C1[:, c_time] = mean_int_per_area_C
            mean_int_per_area_N1[:, c_time] = mean_int_per_area_N
            nuc_Vol1[:, c_time] = nuc_Vol
            cyt_Vol1[:, c_time] = cyt_Vol
            cyt_Vol_sub1[:, c_time] = cyt_Vol_sub
            FL_Conc_T1[:, c_time] = FL_Conc_T
            FL_Conc_C1[:, c_time] = FL_Conc_C
            FL_Conc_N1[:, c_time] = FL_Conc_N
            FL_mean_int_N_thr1[:, c_time] = FL_mean_int_N_thr
            FL_mean_int_C_thr1[:, c_time] = FL_mean_int_C_thr
            Cell_Size1[:, c_time] = cell_size

    with ThreadPoolExecutor(max_workers=16) as executor:
        executor.map(process_timepoint, range(Mask2.shape[2]))

    ALLDATA[0].append(channel)
    ALLDATA[1].append(Cell_Size1)
    ALLDATA[2].append(cell_Vol1)
    ALLDATA[3].append(max_nuc_int1)
    ALLDATA[4].append(mean_cell_Fl1)
    ALLDATA[5].append(Conc_T_cell_Fl1)
    ALLDATA[6].append(mem_area1)
    ALLDATA[7].append(nuc_area1)
    ALLDATA[8].append(cyt_area1)
    ALLDATA[9].append(mean_fl_mem1)
    ALLDATA[10].append(std_fl_mem1)
    ALLDATA[11].append(tot_Fl_mem1)
    ALLDATA[12].append(tot_Fl_cyt1)
    ALLDATA[13].append(tot_Fl_nuc1)
    ALLDATA[14].append(mean_int_per_area_C1)
    ALLDATA[15].append(mean_int_per_area_N1)
    ALLDATA[16].append(nuc_Vol1)
    ALLDATA[17].append(cyt_Vol1)
    ALLDATA[18].append(cyt_Vol_sub1)
    ALLDATA[19].append(FL_Conc_T1)
    ALLDATA[20].append(FL_Conc_C1)
    ALLDATA[21].append(FL_Conc_N1)
    ALLDATA[22].append(FL_mean_int_N_thr1)
    ALLDATA[23].append(FL_mean_int_C_thr1)

save_fold = os.path.join(path_h0, 'FL_extracts', exp_name)
os.makedirs(save_fold, exist_ok=True)
#name3 = f"{exp_folder}_FLEX.mat"
name3 = "_FLEX.mat"
sio.savemat(os.path.join(save_fold, name3), {
    'ALLDATA': ALLDATA,
    'all_back': all_back,
    'I_mean_modifier': I_mean_modifier,
    'peak_cutoff': peak_cutoff,
    'cell_margin': cell_margin
})
