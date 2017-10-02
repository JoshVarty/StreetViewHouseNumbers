import numpy as np
import os
import data_loader
import image_helpers
from digit_struct import DigitStruct


INPUT_ROOT = "../input/"
TRAIN_DIR = os.path.join(INPUT_ROOT, "train")
image_size = 64
num_channels = 3
pixel_depth = 255
num_labels = 5
patch_size_3 = 3
depth = 32

digit_structure_path = os.path.join(TRAIN_DIR, "digitStruct.mat")

#Check if data exists
if not os.path.exists(digit_structure_path):
    data_loader.get_training_data(INPUT_ROOT, "train.tar.gz")

digit_struct = DigitStruct(digit_structure_path)
labels, paths = digit_struct.load_labels_and_paths()

image_paths = [TRAIN_DIR + s for s in paths]

images_normalized = image_helpers.prep_data(image_paths, image_size, num_channels, pixel_depth)


