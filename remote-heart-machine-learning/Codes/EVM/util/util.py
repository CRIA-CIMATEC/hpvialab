"""This module contains simple helper functions """
from __future__ import print_function
import json
import torch
import numpy as np
from PIL import Image
import os
import scipy.io
import cv2

def get_dataset_hr_arrays(dataset_list, ppg_fps):
    """Method to get PPG from json dataset
    Parameters:
    dataset_list (dict) -- dataset json archieve containig  ppg 
    ppg_fps(int)        -- FPS value from heart rate
    """
    result = {}
    for key, value in dataset_list.items():
        result[key] = get_ppg_array(value['PPG'], ppg_fps)
    return result

def get_ppg_array(path, ppg_fps):
    """Method to get ppg array and transform it in a specific size

    Parameters:
    path(str) -- path containing archieve where is ppg
    ppg_fps(int) -- FPS value from heart rate"""

    resize_scale = 30 / ppg_fps
    ppg = scipy.io.loadmat(path)['pulseOxRecord'][0]
    # ppg = np.array(ppg, dtype='uint8')
    
    return cv2.resize(ppg, (1, int(len(ppg)*resize_scale)), interpolation=cv2.INTER_CUBIC)[:,0]
    
def normalize_dataset_hr_array(dataset_hr_array):
    """Method to normalize heart rate array

    Parameters:
    dataset_hr_array(array) -- array containig heart rate values"""
    for key in dataset_hr_array.keys():
        x = dataset_hr_array[key]
        dataset_hr_array[key] = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        values = dataset_hr_array[key]
        dataset_hr_array[key] = dataset_hr_array[key] - np.mean(values)
        dataset_hr_array[key] = dataset_hr_array[key] / np.std(values)
    return dataset_hr_array
    

def denormalize_hr_array(hr_array, path, subject_index):
    """Method to denormalize heart rate array 

    Parameters:
    hr_array(array)         -- array containing heart rate values
    path(str)               -- path where is json
    subject index(int)      -- index value from a subject"""
    f = open(os.path.join(path, "dataset_info.json"))
    dataset_json = json.load(f)
    x = dataset_json[subject_index]
    hr_array = hr_array * (np.amax(x) - np.amin(x)) + np.amin(x)
    return hr_array
    
def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
