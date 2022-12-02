"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    """Method to check if the file extension belongs to a image group"""
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, interval, max_dataset_size=float("inf")):
    """Checks if the directory is valid and returns a list with the path of sorted images"""
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    start, end = interval 
    
    for root, _, fnames in sorted(os.walk(dir)):
        
        fnames.sort()
        for fname in fnames[start: end]:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                
    if max_dataset_size > len(images):
        return images[:len(images)]
    else:
        return images[:max_dataset_size]


def default_loader(path):
    """Method to return image in RGB"""
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):
    """Class to check if a directory exists and have images into the path, 
	also verify image extensions supported """
    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        """Method to apply to transform images if it has the necessity """
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        """Counting number of images into a directory"""
        return len(self.imgs)