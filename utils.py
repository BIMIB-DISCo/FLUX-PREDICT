import pickle
import os
import numpy as np

def write_pickle(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
    return b

def get_experiment_path(path):

    version = 0
    if os.path.isdir(path):
        versions = os.listdir(path)
        version = np.max([int(f.split("_")[1]) for f in versions if f.startswith('version')]) + 1

    version_path = os.path.join(path, "version_" + str(version))
    os.makedirs(version_path)
    return version_path