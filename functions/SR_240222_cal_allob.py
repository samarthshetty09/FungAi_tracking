# -*- coding: utf-8 -*-
"""
Created on Thu May  9 23:46:56 2024

@author: samararth
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio

def cal_allob(ccel, TETC, rang):
    # Initialize all_obj with zeros
    all_obj = np.zeros((ccel, len(TETC)))
    
    # Iterate over each cell index
    for iv in range(ccel):
        for its in rang:
            # Check if TETC[its] is not empty
            if TETC[its].size != 0:
                # Count the occurrences of iv (iv+1 in MATLAB is iv in Python because of zero indexing)
                all_obj[iv, its] = np.sum(TETC[its] == (iv + 1))
            else:
                all_obj[iv, its] = -1
    
    return all_obj


def main():
    # Load an image (should be in grayscale format)
    image_path = '../I2A.tif'  # Specify the path to your image
    image = imageio.imread(image_path)

    # Assuming the image is already in the correct format, if not, convert it

    # Apply the artifact removal function
    cleaned_image = cal_allob(image, 2, 5);

    # Plot the original and cleaned images
    plot_image(image, 'Original Image')
    plot_image(cleaned_image, 'Cleaned Image')

    # Save the cleaned image
    output_path = 'cleaned_image.tif'  # Specify the output path
    imageio.imwrite(output_path, cleaned_image)
    print(f'Cleaned image saved to {output_path}')

if __name__ == '__main__':
    main()