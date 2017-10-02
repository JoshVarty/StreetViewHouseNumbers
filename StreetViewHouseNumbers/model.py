import numpy as np
import os
import data_loader

INPUT_ROOT = "../input/"
digit_structure_path = os.path.join(INPUT_ROOT, "train/digitStruct.mat")

#Check if data exists
if not os.path.exists(digit_structure_path):
    data_loader.get_training_data(INPUT_ROOT, "train.tar.gz")