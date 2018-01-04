import matplotlib.pyplot as plt
import numpy as np

filepath = 'data/npy_256x256/style_val_images.npy'
show_k = 5  # number of images to display

data = np.load(filepath).astype(np.float32)
cmap = 'gray' if len(data.shape) == 3 else None
for i in range(show_k):
    plt.figure()
    plt.imshow(data[i], cmap=cmap)
plt.show()
