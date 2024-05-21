

# -*- coding: utf-8 -*-
"""
Created on Thu May  9 23:50:34 2024

@author: samar
"""

import numpy as np
import matplotlib.pyplot as plt
#import imageio
from PIL import Image

def cal_celldata(all_obj, ccel):
    # Initialize cell_data with zeros, and 5 columns for the data mentioned
    cell_data = np.zeros((ccel, 5))
    
    for iv in range(ccel):
        # Find first and last occurrence where all_obj[iv, :] > 0
        first_occurrence = np.where(all_obj[iv, :] > 0)[0]
        if first_occurrence.size > 0:
            cell_data[iv, 0] = first_occurrence[0] + 1  # +1 to match MATLAB's 1-based indexing
        else:
            cell_data[iv, 0] = np.nan  # Handle case where there is no positive occurrence
        
        last_occurrence = np.where(all_obj[iv, :] > 0)[0]
        if last_occurrence.size > 0:
            cell_data[iv, 1] = last_occurrence[-1] + 1  # +1 to match MATLAB's 1-based indexing
        else:
            cell_data[iv, 1] = np.nan  # Handle case where there is no positive occurrence
    
    for iv in range(ccel):
        if not np.isnan(cell_data[iv, 0]) and not np.isnan(cell_data[iv, 1]):
            cell_data[iv, 2] = cell_data[iv, 1] - cell_data[iv, 0] + 1
            # Convert MATLAB's 1-based to Python's 0-based index for slicing
            aa1 = all_obj[iv, :]
            aa2 = aa1[int(cell_data[iv, 0] - 1):int(cell_data[iv, 1])]
            aa3 = np.where(aa2 == 0)[0]
            cell_data[iv, 3] = len(aa3)  # Number of times it disappears
            cell_data[iv, 4] = (cell_data[iv, 3] * 100) / cell_data[iv, 2]  # Percentage of times it disappears
        else:
            cell_data[iv, 2:] = np.nan  # Handle cases with no occurrences
    
    return cell_data


def plot_image(image, title):
    plt.figure()
    plt.imshow(image)  # Use the gray colormap for grayscale images
    plt.title(title)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

def main():
    # Load an image (should be in grayscale format)
    image_path = '../I2A.tif'  # Specify the path to your image
    image = Image.open(image_path)
    img_array = np.array(image)
    print(img_array)
    # Assuming the image is already in the correct format, if not, convert it

    # Apply the artifact removal function
    op = cal_celldata(img_array, 3);

    print(op)

if __name__ == '__main__':
    main()