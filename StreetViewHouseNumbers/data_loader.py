import sys
import pickle
import os
from six.moves.urllib.request import urlretrieve
import tarfile

def savePickle(object, filePath):
    with open(filePath, 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

def openPickle(filepath): 
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except:
        return None

def get_training_data(dest_folder, dest_filename):
    url = "http://ufldl.stanford.edu/housenumbers/train.tar.gz"

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    dest_file_path = os.path.join(dest_folder, dest_filename)
    print('Attempting to download:', url) 
    filename, _ = urlretrieve(url, dest_file_path, reporthook=download_progress_hook)
    print('\nDownload Complete!')
    extract(dest_file_path, dest_folder)


def extract(tar_filepath, dest_folder):
    print("Extracting tarfile...", tar_filepath)
    tar = tarfile.open(tar_filepath)
    sys.stdout.flush()
    tar.extractall(dest_folder)
    tar.close()


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

