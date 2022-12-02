from .base_dataset import BaseDataset, get_params, get_transform
import torch
import numpy as np


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, dataset_list, length):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        
        BaseDataset.__init__(self, opt, dataset_list, length)
        
        # self.feature_image_list = make_dataset(self.feature_image_path, opt.max_dataset_size)  # get image paths
        
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image

        self.input_nc = self.opt.input_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random tuple for data indexing - (fist value is a string containing the name of the subject and 
            the second value is an integer representing the second of the video

        Returns a dictionary that contains feature_image, hr_value, subject_index, feature_image_path and hr_value_path
            feature_image (tensor) - - feature_image
            hr_value (tensor) - - its corresponding hr value
            subject_index (str) - - string representing the subject
        """
        subject_index, target_index = index
        
        feature_image_list = self.feature_image_list[subject_index]["feature_image"]
        feature_image_path = feature_image_list[target_index]
        
        hr_value_path = self.feature_image_list[subject_index]
        
        hr_value_array = self.hr_value_array[subject_index]               
        hr_value = hr_value_array[target_index * 30 : (target_index + 1) * 30]

        hr_value = torch.FloatTensor(np.array(hr_value)) #  self.quantify(torch.FloatTensor(hr_value))
        
        feature_image = np.load(feature_image_path)

        
        feature_image_transform_params = get_params(self.opt, feature_image.shape)
        feature_image_transform = get_transform(self.opt, feature_image_transform_params, grayscale = (self.input_nc == 1))
           

        feature_image = feature_image_transform(feature_image)
        
        
        return {'feature_image': feature_image, 'ground_truth': hr_value, 'subject_index': subject_index, 'feature_image_paths': feature_image_path, 'hr_paths': hr_value_path}
        
    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.length