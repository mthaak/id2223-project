import os
import shutil

import numpy as np

sample_frac = 0.1
input_dirname = 'data/wikiart_csv'
output_dirname = 'data/wikiart_sample_csv'

# Create output directory if not exists
if not os.path.exists(output_dirname):
    os.makedirs(output_dirname)


def create_one_sample(input_filename, output_filename):
    data = np.loadtxt(input_filename, dtype=str)
    np.random.shuffle(data)
    sample_size = int(sample_frac * len(data))
    data_sample = data[0: sample_size]
    np.savetxt(output_filename, data_sample, fmt='%s')


# Create samples
create_one_sample(input_dirname + '/artist_train.csv', output_dirname + '/artist_train.csv')
create_one_sample(input_dirname + '/artist_val.csv', output_dirname + '/artist_val.csv')
create_one_sample(input_dirname + '/genre_train.csv', output_dirname + '/genre_train.csv')
create_one_sample(input_dirname + '/genre_val.csv', output_dirname + '/genre_val.csv')
create_one_sample(input_dirname + '/style_train.csv', output_dirname + '/style_train.csv')
create_one_sample(input_dirname + '/style_val.csv', output_dirname + '/style_val.csv')

# Copy class files
shutil.copy(input_dirname + '/artist_class.txt', output_dirname)
shutil.copy(input_dirname + '/genre_class.txt', output_dirname)
shutil.copy(input_dirname + '/style_class.txt', output_dirname)
