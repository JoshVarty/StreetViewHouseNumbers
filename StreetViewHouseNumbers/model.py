import numpy as np
import os
import data_loader
from digit_struct import DigitStruct

INPUT_ROOT = "../input/"
digit_structure_path = os.path.join(INPUT_ROOT, "train/digitStruct.mat")

#Check if data exists
if not os.path.exists(digit_structure_path):
    data_loader.get_training_data(INPUT_ROOT, "train.tar.gz")

digit_struct = DigitStruct(digit_structure_path)
labels, paths = digit_struct.load_labels_and_paths()

print(labels.shape)
print(paths.shape)
