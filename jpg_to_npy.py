import glob

from skimage import io
import numpy as np

images_files = glob.glob('data/*.jpg')

n = len(image_files)
height = 227
width = 227
channels = 3
output_filename = 'images.npy'

images = np.empty((n, height, width, channels))

for i, images_file in enumerate(images_files):
    image_data = io.imread(images_file)
    images[i] = image_data

np.save(output_filename, images)
