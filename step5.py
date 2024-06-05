# -*- coding: utf-8 -*-
"""
Created on Sat May 11 23:38:43 2024

@author: samarth
"""
import numpy as np
import os
import statistics
import time
from scipy.io import savemat
import matplotlib.pyplot as plt
from functions.OAM_230919_remove_artif import remove_artif
from functions.OAM_231216_bina import binar
from PIL import Image
#import cupy as cp
from skimage.morphology import thin



#Global Variables
input_images = "/Users/samarth/Documents/MirandaLabs/New_tracks/"
sav_path = "/Users/samarth/Documents/MirandaLabs/New_tracks/res/"
pos = 'pos0_2'

#Helper Functions
def key(filename):
    return int(filename.split('_')[1])

def display_image(image, title="Image"):
    if image.ndim > 2:
        image = np.squeeze(image)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(image, interpolation='none')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def OAM_23121_tp3(M, cel, no_obj1, A):
    tp3 = np.array(M)  # Ensure M is a numpy array
    tp3[tp3 == cel] = no_obj1 + A
    return tp3
    
    
"""
1. Init loaded Variables
"""

image_files = [file for file in os.listdir(input_images) if file.endswith('_ART_masks.tif')]
sorted_files = sorted(image_files, key=key)
mask_paths = [os.path.join(input_images, img_name) for img_name in sorted_files]

"""
 2. load first mask and allocate 
"""   
#m0 = PIL.Image.open(mask_paths[0]).convert("L") # zero start #
m0 = Image.fromarray(np.array(Image.open(mask_paths[0])).astype("uint16")) #LOAD AS UINT16
IS1 = np.array(m0) # image to array # plt.imshow(IS1)
IblankG=np.zeros(IS1.shape, dtype="uint16") # allocate
masks = np.uint16(np.zeros((IS1.shape[0], IS1.shape[1], len(mask_paths)))) # allocate
masks[:,:,0]= IS1


IblankG = np.zeros_like(IS1, dtype=np.uint16)
glLen = len(mask_paths) - 1
for it0 in range(1,len(mask_paths)):
    print(f'it0={it0}')

    # Loading future cellpose mask
    m0b =  Image.fromarray(np.array(Image.open(mask_paths[it0])).astype("uint16"))
    print(f"Processing image: {mask_paths[it0]}")
    #IS2 = masks[:, :, 0].astype(np.uint16) #Shape of IS2 = 126*692 if im_no1-1 is taken
    IS2 = np.array(m0b)
    
    print("Shape of IS2:", IS2.shape)
    print("Number of dimensions in IS2:", IS2.ndim)

    IS2 = remove_artif(IS2)
    #display_image(IS2)

    IS2C = IS2
    IS1B = binar(IS1)
    # The multiplication
    IS3 = np.multiply(IS1B,IS2)
    
    tr_cells = np.unique(IS1[IS1 != 0])
    gap_cells = np.unique(IblankG[IblankG != 0])
    cells_tr = np.concatenate((tr_cells, gap_cells))
    
    Iblank0 = np.zeros_like(IS1, dtype=np.uint16)

    # Tracking cells across images
    if cells_tr.size != 0:
        for it1 in sorted(cells_tr):
            IS5 = (IS1 == it1)
            IS5=IS5.astype(np.uint16)
            IS56 = thin(IS5,1) # plt.imshow(IS56)  #  sum(sum(IS56))
            IS6A = np.multiply(IS56,IS3)
    
            if np.sum(IS5) == 0:
                IS5 = (IblankG == it1)
                IS6A = np.multiply(IS56,IS2C)
                IblankG[IblankG == it1] = 0
    
            # Update masks to avoid overlapping cells
            if np.sum(IS6A) != 0:
                IS2ind = np.bincount(IS6A.astype(int).flat)[1:].argmax() + 1
                #IS2ind=(statistics.mode(IS6A[np.nonzero(IS6A)]))
                Iblank0[IS2 == IS2ind] = it1
                IS3[IS3 == IS2ind] = 0
                IS2C[IS2 == IS2ind] = 0
        

    # Detect segmentation gaps
    seg_gap = np.setdiff1d(tr_cells, np.unique(Iblank0))
    if seg_gap.size != 0:
        for itG in seg_gap:
            IblankG[IS1 == itG] = itG

    # Detect new cells entering the frame
    Iblank0B = np.copy(Iblank0)     
    Iblank0B[np.nonzero(Iblank0B)] = 1;   # plt.imshow(Iblank0B)
    Iblank0B = (Iblank0B < 0.5).astype(np.uint16)
    #ISB = IS2 * (~Iblank0B).astype(np.uint16)
    ISB = np.multiply(IS2,Iblank0B)
    newcells = np.unique(ISB)
    Iblank=Iblank0;
    newcells = newcells[1:len(newcells)]
    A = 1
    if newcells.size != 0:
        """
        if cells_tr.size > 0:
            next_index = np.max(cells_tr) + A
        else:
            next_index = A  # Adjust this depending on how you want to handle this edge case
        """
        for it2 in newcells:
            Iblank[IS2 == it2] = max(cells_tr)+A;
            A += 1

    masks[:, :, it0] = Iblank
    IS1 = masks[:, :, it0]



for it4 in range(1,glLen): 
    plt.imshow(masks[:,:,it4])

"""
Tracks as a tensor
"""

im_no = masks.shape[2]
# Find all unique non-zero cell identifiers across all time points
ccell2 = np.unique(masks[masks != 0])
# Initialize Mask2 with zeros of the same shape as masks
Mask2 = np.zeros_like(masks)


#TODO: instead of np use cpypy

# Process each unique cell ID
for itt3 in range(len(ccell2)):
    cell_id = ccell2[itt3]
    # Find all indices where the current cell ID is present
    pix3 = np.where(masks == cell_id)
    # Assign a new sequential ID to these positions in Mask2
    Mask2[pix3] = itt3 + 1  # +1 to make IDs 1-based like MATLAB if needed

    # Optional: print the current iteration's progress
    print("Processing cell ID:", itt3 + 1, "out of", len(ccell2))


"""
Get Cell Presence
"""

Mask3 = Mask2.copy()
numbM = im_no  # Number of time points, should be defined previously as masks.shape[2]

obj = np.uint16((np.unique(Mask3)))  # Unique cell identifiers


no_obj1 = int(np.max(obj))  # Maximum cell identifier

# Timing start
start_time = time.time()

# Initialize the presence matrix tp_im with zeros
tp_im = np.zeros((no_obj1, numbM), dtype=np.float64)

print(tp_im)

# Loop over each cell to determine its presence across time points
for cel in range(1, no_obj1 + 1):  # Python's range is exclusive, so +1 to include no_obj1
    Ma = (Mask3 == cel)  # Create a mask for the current cell
    for ih in range(numbM):  # Loop over each time point
        if np.sum(Ma[:, :, ih]) != 0:  # Check if the cell is present at this time point
            tp_im[cel - 1, ih] = 1  # Update tp_im, -1 for 0-based indexing in Python

    # Optional: Print the current cell being processed
    print(f"Processing cell: {cel}")

# Print elapsed time
print(f"Elapsed time: {time.time() - start_time} seconds.")

"""
data = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 1]
]

# Convert the list of lists to a NumPy array of float64 type
tp_im = np.array(data, dtype=np.float64)
"""

"""
Split Inturrupted Time Series
"""
start_time = time.time()
#numbM -= 1
numbM = Mask3.shape[2] - 1;
for cel in range(0, np.max(obj)):
    print(cel)
    tp_im2 = np.diff(tp_im[cel, :]);

    tp1 = np.where(tp_im2 == 1)[0]
    tp2 = np.where(tp_im2 == -1)[0]

    maxp = np.uint16(np.sum(np.sum(Mask3[:, :, numbM] == cel)))

    if len(tp1) == 1 and len(tp2) == 1 and maxp != 0:  # has one interruption
        for itx in range(tp1[0], numbM):
            tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
            Mask3[:, :, itx] = tp3
        no_obj1 += A

    elif len(tp1) == 1 and len(tp2) == 1 and maxp == 0:  # has one interruption
        pass

    elif len(tp1) == len(tp2) + 1 and maxp != 0:
        tp2 = np.append(tp2, numbM)
        for itb in range(1, len(tp1)):
            for itx in range(tp1[itb] + 1, tp2[itb] + 1):
                tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                Mask3[:, :, itx] = tp3
            no_obj1 += A

    elif len(tp1) == 0 or len(tp2) == 0:  # it's a normal cell, it's born and stays until the end
        pass

    elif len(tp1) == len(tp2):
        if tp1[0] > tp2[0]:
            tp2 = np.append(tp2, numbM)
            for itb in range(len(tp1)):
                for itx in range(tp1[itb] + 1, tp2[itb + 1] + 1):
                    tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                    Mask3[:, :, itx] = tp3
                no_obj1 += A
        elif tp1[0] < tp2[0]:
            for itb in range(1, len(tp1)):
                for itx in range(tp1[itb] + 1, tp2[itb] + 1):
                    tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                    Mask3[:, :, itx] = tp3
                no_obj1 += A
        elif len(tp2) > 1:
            for itb in range(1, len(tp1)):
                for itx in range(tp1[itb] + 1, tp2[itb] + 1):
                    tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                    Mask3[:, :, itx] = tp3
                no_obj1 += A

print(f"Time taken: {time.time() - start_time} seconds")

plt.figure()
plt.imshow(tp_im, aspect='auto', cmap='viridis')
plt.title("Cell Presence Over Time")
plt.xlabel("Time")
plt.ylabel("Cells")
plt.show()

"""
start_time = time.time()

numbM = Mask3.shape[2]  # Should match the MATLAB 'numbM'

for cel in range(1, int(np.max(obj)) + 1):
    print(cel)
    tp_im2 = np.diff(tp_im[cel - 1, :], axis=0)

    tp1 = np.where(tp_im2 == 1)[0]
    tp2 = np.where(tp_im2 == -1)[0]

    maxp = np.sum(Mask3[:, :, numbM - 1] == cel)

    if len(tp1) == 1 and len(tp2) == 1 and maxp != 0:  # has one interruption
        for itx in range(tp1[0], numbM):
            tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
            Mask3[:, :, itx] = tp3
        no_obj1 += A

    elif len(tp1) == 1 and len(tp2) == 1 and maxp == 0:  # has one interruption and ends
        pass

    elif len(tp1) == len(tp2) + 1 and maxp != 0:  # has one more tp1
        tp2 = np.append(tp2, numbM - 1)
        for itb in range(1, len(tp1)):
            for itx in range(tp1[itb] + 1, tp2[itb] + 1):
                tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                Mask3[:, :, itx] = tp3
        no_obj1 += A

    elif len(tp1) == len(tp2):  # Normal cell or has one interruption
        if len(tp1) > 0 and len(tp2) > 0:  # Check if tp1 and tp2 are not empty
            if tp1[0] > tp2[0]:
                tp2 = np.append(tp2, numbM - 1)
                for itb in range(0, len(tp1)):
                    for itx in range(tp1[itb] + 1, tp2[itb + 1] + 1):
                        tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                        Mask3[:, :, itx] = tp3
                no_obj1 += A
            elif tp1[0] < tp2[0]:
                for itb in range(1, len(tp1)):
                    for itx in range(tp1[itb] + 1, tp2[itb] + 1):
                        tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                        Mask3[:, :, itx] = tp3
                no_obj1 += A
        elif len(tp2) > 1:  # If it has multiple interruptions
            for itb in range(1, len(tp1)):
                for itx in range(tp1[itb] + 1, tp2[itb] + 1):
                    tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                    Mask3[:, :, itx] = tp3
            no_obj1 += A

print(f"Elapsed time: {time.time() - start_time} seconds.")
"""

"""
Get Cell Presence 2
"""
numbM = Mask3.shape[2]
obj = np.unique(Mask3)

tp_im = np.zeros((obj.max(), numbM))

start_time = time.time()

for cel in range(1, obj.max() + 1):
    Ma = (Mask3 == cel)
    for ih in range(numbM):
        if np.sum(Ma[:, :, ih]) != 0:
            tp_im[cel - 1, ih] = 1

end_time = time.time()
print(f"Elapsed time: {end_time - start_time:.6f} seconds")

# Get good cells
cell_artifacts = np.zeros(tp_im.shape[0])

for it05 in range(tp_im.shape[0]):
    arti = np.where(np.diff(tp_im[it05, :]) == -1)[0]
    if arti.size != 0:
        cell_artifacts[it05] = it05

goodcells = np.setdiff1d(np.arange(tp_im.shape[0]), cell_artifacts)

print("Good cells:", goodcells)

#####
"""
Track as Tensor 2
"""

im_no = Mask3.shape[2]
Mask4 = np.zeros_like(masks)  # Initialize Mask4 with zeros

for itt3 in range(len(goodcells)):  # Iterate over good cells
    pix3 = np.where(Mask3 == goodcells[itt3])  # Find indices of good cells in Mask3
    Mask4[pix3] = itt3 + 1  # Re-assign ID

print("Updated Mask4:")
print(Mask4)


for it4 in range(1,glLen): 
    plt.imshow(Mask4[:,:,it4])

"""
Get Cell Presence 3
"""
Mask5 = Mask4.copy()
numbM = Mask4.shape[2]
obj = np.unique(Mask4)

start_time = time.time()

tp_im = np.zeros((obj.max(), numbM))

for cel in range(1, obj.max() + 1):
    Ma = (Mask5 == cel)
    for ih in range(numbM):
        if np.sum(Ma[:, :, ih]) != 0:
            tp_im[cel - 1, ih] = 1

end_time = time.time()
print(f"Elapsed time: {end_time - start_time:.6f} seconds")


for it4 in range(1,glLen): 
    plt.imshow(Mask5[:,:,it4])


# Calculate first and last detection
cell_exists0 = np.zeros((2, tp_im.shape[0]))

for itt2 in range(tp_im.shape[0]):
    cell_exists0[0, itt2] = np.argmax(tp_im[itt2, :] != 0)
    cell_exists0[1, itt2] = len(tp_im[itt2, :]) - 1 - np.argmax(tp_im[itt2, ::-1] != 0)

# Sort the cells by their first detection time
sortOrder, _ = zip(*sorted(enumerate(cell_exists0[0, :]), key=lambda x: x[1]))
sortOrder = np.array(sortOrder)

cell_exists = cell_exists0[:, sortOrder]

# Re-label
Mask6 = np.zeros_like(Mask5)

for itt3 in range(len(sortOrder)):
    pix3 = np.where(Mask5 == sortOrder[itt3] + 1)  # Adding 1 to match MATLAB's 1-based indexing
    Mask6[pix3] = itt3 + 1  # Assign new labels

print("Updated Mask6:")
print(Mask6)


for it4 in range(1,glLen): 
    plt.imshow(Mask6[:,:,it4])


"""
Get Cell Presence 3-2
"""
Mask7 = Mask6.copy()
numbM = Mask6.shape[2]
obj = np.unique(Mask6)

# Visualize timeseries of masks (view with caution)
plt.figure()
plt.imshow(np.concatenate(masks, axis=1))  # Combining masks for visualization
plt.title("Timeseries of Masks")
plt.show()

no_obj1 = obj.max()
A = 1

start_time = time.time()

tp_im = np.zeros((obj.max(), numbM))

for cel in range(1, obj.max() + 1):
    Ma = (Mask7 == cel)
    for ih in range(numbM):
        if np.sum(Ma[:, :, ih]) != 0:
            tp_im[cel - 1, ih] = 1

end_time = time.time()
print(f"Elapsed time: {end_time - start_time:.6f} seconds")


for it4 in range(1,glLen): 
    plt.imshow(Mask7[it4])


# Visualization of tp_im
plt.figure()
plt.imshow(tp_im, aspect='auto', cmap='viridis')
plt.title("Cell Presence Over Time")
plt.xlabel("Time")
plt.ylabel("Cells")
plt.show()


#####
plt.figure(figsize=(12, 6))
plt.imshow(tp_im, aspect='auto', cmap='viridis')
plt.colorbar(label='Pixel Count')
plt.xlabel('Image Number (Time)')
plt.ylabel('Object Number')
plt.title('tp_im')
plt.show()   


""" 
Calculate Size
"""
no_obj = np.max(np.unique(Mask3))
im_no = Mask3.shape[2]
all_ob = np.zeros((no_obj, im_no))

start_time = time.time()

for ccell in range(1, no_obj + 1):
    Maa = (Mask3 == ccell)
    for io in range(im_no):
        pix = np.sum(Maa[:, :, io])
        all_ob[ccell - 1, io] = pix

end_time = time.time()
print(f"Elapsed time: {end_time - start_time:.6f} seconds")

# Visualization of all_ob
plt.figure()
plt.imshow(all_ob, aspect='auto', cmap='viridis')
plt.title("Cell Sizes Over Time")
plt.xlabel("Time")
plt.ylabel("Cells")
plt.show()

for it4 in range(1,glLen): 
    plt.imshow(Mask7[:,:,it4])

####
# Calculate Size
uniq = np.unique(Mask7)
no_obj2 = int(np.max(np.unique(Mask7)))
all_ob = np.zeros((no_obj2, im_no))

# Process Mask3 and calculate pixel sums
start_time = time.time()
for ccell in range(no_obj2):
    Maa = (Mask7
           == (ccell + 1))
    for x in range(im_no):
        pix = np.sum(Maa[:, :, x])
        all_ob[ccell, x] = pix
        

for it4 in range(1,glLen): 
    plt.imshow(Mask7[:,:,it4])

        
# Visualization of all_ob
plt.figure(figsize=(12, 6))
plt.imshow(all_ob, aspect='auto', cmap='viridis')
plt.colorbar(label='Pixel Count')
plt.xlabel('Image Number (Time)')
plt.ylabel('Object Number')
plt.title('Pixel Counts of Objects Over Time')
plt.show()        
        
plt.figure(figsize=(12, 6))
for ccell in range(no_obj2):
    plt.plot(all_ob[ccell], label=f'Object {ccell + 1}')

plt.xlabel('Image Number (Time)')
plt.ylabel('Pixel Count')
plt.title('Pixel Counts of Objects Over Time')
plt.legend()
plt.show()
        
print("Elapsed time:", time.time() - start_time)

# Find the first and last non-zero values in all_ob
cell_exists = np.zeros((2, all_ob.shape[0]))

for itt2 in range(all_ob.shape[0]):
    non_zero_indices = np.nonzero(all_ob[itt2, :])[0]
    if non_zero_indices.size > 0:
        cell_exists[:, itt2] = [non_zero_indices[0] + 1, non_zero_indices[-1] + 1]

# Convert Mask3 to a list of 2D arrays
Mask3 = [Mask3[:, :, its] for its in range(Mask3.shape[2])]

# Save the results
savemat(f"{sav_path}{pos}_ART_Track.mat", {
    'all_ob': all_ob,
    'Mask3': Mask3,
    'no_obj': no_obj2,
    'im_no': im_no,
    'cell_exists': cell_exists,
    # Add other variables as needed
})







    
    







