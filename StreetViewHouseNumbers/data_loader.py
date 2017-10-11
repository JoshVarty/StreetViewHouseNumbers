import sys
import pickle
import os
from six.moves.urllib.request import urlretrieve
import tarfile
import scipy.io
import numpy as np

def savePickle(object, filePath):
    with open(filePath, 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

def openPickle(filepath): 
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except:
        return None

def load_data():
    INPUT_ROOT = "../input/"
    TRAIN_DIR = os.path.join(INPUT_ROOT, "train/")

    pickle_path = os.path.join(TRAIN_DIR, "data.pickle")
    data = openPickle(pickle_path)
    if data != None:
        return data

    train_mat_path = os.path.join(TRAIN_DIR, "train_32x32.mat")
    extra_mat_path = os.path.join(TRAIN_DIR, "extra_32x32.mat")

    #Check if mat files exist
    if not os.path.exists(train_mat_path):
        download_data(TRAIN_DIR, "train_32x32.mat")

    if not os.path.exists(extra_mat_path):
        download_data(TRAIN_DIR, "extra_32x32.mat")

    train_data = scipy.io.loadmat(train_mat_path, variable_names='X').get('X')
    train_labels = scipy.io.loadmat(train_mat_path, variable_names='y').get('y')
    extra_data = scipy.io.loadmat(extra_mat_path, variable_names='X').get('X')
    extra_labels = scipy.io.loadmat(extra_mat_path, variable_names='y').get('y')

    train_data = train_data.transpose(3,0,1,2)
    extra_data = extra_data.transpose(3,0,1,2)

    print("Raw training data", train_data.shape, train_labels.shape)
    print("Extra training data", extra_data.shape, extra_labels.shape)
    print()

    train_labels[train_labels == 10] = 0
    extra_labels[extra_labels == 10] = 0

    #We want a validation set based exclusively on our target images from train_data
    #We also want an even distribution of classes in it
    n_labels = 10
    valid_index = np.zeros(train_labels.shape[0], dtype=bool)

    for i in np.arange(n_labels):
        valid_index[(np.where(train_labels[:,0] == (i))[0][:600].tolist())] = True

    valid_data = train_data[valid_index, :, :, :]
    valid_labels = train_labels[valid_index]

    train_labels = train_labels[~valid_index]
    train_data = train_data[~valid_index,:,:,:]
    train_labels = np.concatenate((train_labels, extra_labels))
    train_data = np.concatenate((train_data, extra_data))

    print("Valid Set", valid_data.shape)
    print("Train Set", train_data.shape)

    np.random.seed(42)
    def randomize(dataset, labels):
      permutation = np.random.permutation(labels.shape[0])
      shuffled_dataset = dataset[permutation,:,:,:]
      shuffled_labels = labels[permutation]
      return shuffled_dataset, shuffled_labels

    valid_data, valid_labels= randomize(valid_data, valid_labels)
    train_data, train_labels= randomize(train_data, train_labels)

    train_data = im2gray(train_data)[:,:,:,0]
    valid_data = im2gray(valid_data)[:,:,:,0]

    train_data = GCN(train_data)
    valid_data = GCN(valid_data)
    
    train_data = train_data.reshape((train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))
    valid_data = valid_data.reshape((valid_data.shape[0], valid_data.shape[1], valid_data.shape[2], 1))

    savePickle((train_data, train_labels, valid_data, valid_labels), os.path.join(pickle_path))
    return (train_data, train_labels, valid_data, valid_labels)


def im2gray(image):
    # http://www.eyemaginary.com/Rendering/TurnColorsGray.pdf
    image = image.astype(float)
    image_gray = np.dot(image, [[0.2989],[0.5870],[0.1140]])
    return image_gray

def GCN(image, min_divisor=1e-4):
    imsize = image.shape[0]
    mean = np.mean(image, axis=(1,2), dtype=float)
    std = np.std(image, axis=(1,2), dtype=float, ddof=1)
    std[std < min_divisor] = 1.
    image_GCN = np.zeros(image.shape, dtype=float)
    
    for i in np.arange(imsize):
        image_GCN[i,:,:] = (image[i,:,:] - mean[i]) / std[i]
        
    return image_GCN


def download_data(dest_folder, filename):
    url = "http://ufldl.stanford.edu/housenumbers/" + filename

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    dest_file_path = os.path.join(dest_folder, filename)
    print('Attempting to download:', url) 
    filename, _ = urlretrieve(url, dest_file_path, reporthook=download_progress_hook)
    print('\nDownload Complete!')


last_percent_reported = None
def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)
  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
    last_percent_reported = percent

