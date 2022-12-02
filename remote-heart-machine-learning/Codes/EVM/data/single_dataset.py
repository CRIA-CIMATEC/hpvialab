import numpy as np
from .base_dataset import BaseDataset, get_params, get_transform

class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.
    """

    def __init__(self, opt, dataset_list, length):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt, dataset_list, length)
        # self.feature_image_list = make_dataset(self.feature_image_path, opt.max_dataset_size)
        self.transform = get_transform(opt, grayscale=(self.opt.input_nc == 1))

        self.input_nc = self.opt.input_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        subject_index, target_index = index
        
        
        feature_image_list = self.feature_image_list[subject_index]["feature_image"]
        feature_image_path = feature_image_list[target_index]
        # feature_image = Image.open(feature_image_path).convert('RGB')
        # feature_image = self.transform(feature_image)
        feature_image = np.load(feature_image_path)

        feature_image_transform_params = get_params(self.opt, feature_image.shape)
        feature_image_transform = get_transform(self.opt, feature_image_transform_params, grayscale = (self.input_nc == 1))

        feature_image = feature_image_transform(feature_image)

        return {'feature_image': feature_image, 'subject_index': subject_index, 'feature_image_paths': feature_image_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.feature_image_path)
