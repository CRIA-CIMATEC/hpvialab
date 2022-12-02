import glob
import os
import pickle

from os.path import join, isfile
import cv2
import numpy as np

import src.datasets.dataset_utils as du
from src.constants import Constants as c
from src.datasets.torch_helper import ECGCachedWindowsDataset


class GenericConstants:
    glob_to_pickled_data: str = join("generic", c.sample_name, "*.pkl")
    path_to_cache: str = join(c.cache_base_path, "generic", c.sample_name)
    window_size: int = 2560


def iterate_generic(basepath: str):
    pikls = glob.glob(basepath + GenericConstants.glob_to_pickled_data)
    for pf in pikls:
        with open(pf, 'rb') as f:
            blob = pickle.load(f)
            ecg = blob['ECG']
            lbls = blob['label']
            identifier = blob['identifier']
            yield identifier, ecg, lbls


def iterate_clean_labeled_sections(ecg, lbls):
    l_value = lbls[0]
    l_index = 0
    for i in range(1, len(lbls)):
        i_val = lbls[i]
        if i_val != l_value:
            yield l_value, ecg[l_index:i]
            l_value = i_val
            l_index = i
    yield l_value, ecg[l_index:]


def downsample(ecg):
    sampled = cv2.resize(ecg.astype(np.float), (1, int((ecg.shape[0] / 700.) * 256.)),
                        interpolation=cv2.INTER_LINEAR)
    return sampled


def load_ecg_windows(basepath: str):
    def make_windows(d_ecg_array):
        ws = GenericConstants.window_size
        max_len = du.get_max_len(d_ecg_array, ws)
        windows = du.make_windows_list(d_ecg_array, max_len, ws)
        return windows

    windows = []
    window_labels = []
    identifiers = []
    for identifier, ecg, lbls in iterate_generic(basepath):
        mean, std = du.get_mean_std(ecg)
        ecg = du.normalize(ecg, mean, std)
        for section_lbl, section_ecg in iterate_clean_labeled_sections(ecg, lbls):
            # section_ecg = downsample(section_ecg)
            ws = make_windows(section_ecg)
            windows += ws
            window_labels += [section_lbl]*len(ws)
            identifiers += [identifier]*len(ws)

    if windows != []:
        du.create_path_if_needed(GenericConstants.path_to_cache)
        with open(join(GenericConstants.path_to_cache, 'identifiers.npy'), 'wb') as f:
            pickle.dump(identifiers, f)

    return windows, window_labels


class ECGGenericCachedWindowsDataset(ECGCachedWindowsDataset):

    def __init__(self, basepath: str):
        super(ECGGenericCachedWindowsDataset, self).__init__(basepath, GenericConstants.path_to_cache, load_ecg_windows)
        # self.window_files = os.listdir(GenericConstants.path_to_cache)

    def get_item(self, idx):
        labels = None
        # else we assume it is a single index so:
        with open(join(GenericConstants.path_to_cache, f'window-{idx}.data.npy'), 'rb') as f:
            sample = np.load(f)

        with open(join(GenericConstants.path_to_cache, 'identifiers.npy'), 'rb') as f:
            identifiers = pickle.load(f)[idx]

        if isfile(join(GenericConstants.path_to_cache, f'window-{idx}.label.npy')):
            with open(join(GenericConstants.path_to_cache, f'window-{idx}.label.npy'), 'rb') as f:
                labels = pickle.load(f)

        return sample, labels, identifiers

    @property
    def augmentations(self):
        return []