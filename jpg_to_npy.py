import glob

import numpy as np
from skimage import io

image_files = glob.glob('data/*.jpg')

n = len(image_files)
height = 227
width = 227
channels = 3
output_filename = 'images.npy'

images = np.empty((n, height, width, channels))

for i, image_file in enumerate(image_files):
    image_data = io.imread(image_file)
    images[i] = image_data

np.save(output_filename, images)
