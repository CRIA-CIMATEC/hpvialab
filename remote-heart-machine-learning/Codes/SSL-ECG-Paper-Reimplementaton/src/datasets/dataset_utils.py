import numpy as np
import pickle
import os
from pathlib import Path


def get_max_len(data, window_len):
    max_len = (len(data) // window_len) * window_len
    return max_len


def make_windows_list(data, max_len, window_len):
    d = data[:max_len]
    n_windows = len(data) // window_len
    # print(len(data), len(d), n_windows, max_len, window_len)
    mat = d.reshape((n_windows, window_len))
    w = list(mat)
    return w


def normalize(data, data_mean, data_std):
    data_scaled = (data - data_mean) / data_std
    return data_scaled


def get_mean_std(all_data):
    all_data = np.sort(all_data)
    data_mean = np.mean(all_data)
    data_std = np.std(all_data[np.int(0.025 * len(all_data)): np.int(0.975 * len(all_data))])
    return data_mean, data_std


def save_window_to_cache(path_to_cache, window, window_label, identifier):
    with open(os.path.join(path_to_cache, f'window-{identifier}.data.npy'), 'wb') as f:
        np.save(f, window)
    with open(os.path.join(path_to_cache, f'window-{identifier}.label.npy'), 'wb') as f:
        pickle.dump(window_label, f)


def save_windows_to_cache(path_to_cache, windows, window_labels):
    print(f'saving {len(windows)}  and {len(window_labels)} to cache')
    for i, (w, l) in enumerate(zip(windows, window_labels)):
        save_window_to_cache(path_to_cache, w, l, i)


def cache_is_empty(path_to_cache):
    if not os.path.exists(path_to_cache): return True # non existent paths are empty
    path_list = os.listdir(path_to_cache)
    if '.DS_Store' in path_list: # some macOS specific fix
        path_list.remove('.DS_Store')
    is_empty = len(path_list) == 0
    return is_empty


def create_path_if_needed(path):
    Path(path).mkdir(parents=True, exist_ok=True)