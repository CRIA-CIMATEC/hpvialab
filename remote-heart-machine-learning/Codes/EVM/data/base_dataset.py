"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

from abc import ABC, abstractmethod
import math

try:
	from util import util
except:
	from EVM.util import util

class BaseDataset(data.Dataset, ABC):
	"""This class is an abstract base class (ABC) for datasets.

	To create a subclass, you need to implement the following four functions:
	-- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
	-- <__len__>:                       return the size of dataset.
	-- <__getitem__>:                   get a data point.
	-- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
	"""

	def __init__(self, opt, dataset_list, length):
		"""Initialize the class; save the options in the class

		Parameters:
			opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""
		self.opt = opt

		self.phase = opt.phase
		self.feature_image_path = opt.feature_image_path
		# TODO: mudar nome de feature_image_list para json_dataset ou afim
		self.feature_image_list = dataset_list
		self.length = length
		if self.phase != 'test':
			self.hr_value_array = util.get_dataset_hr_arrays(dataset_list, self.opt.ppg_fps)
			self.hr_value_array = util.normalize_dataset_hr_array(self.hr_value_array)

	# def get_hr_array(self, path):
		# mat = scipy.io.loadmat(path)['bpm'][0]
		# mat = (mat - 40.0) / (240.0 - 40.0)
		# return mat 


	@staticmethod
	def modify_commandline_options(parser, is_train):
		"""Add new dataset-specific options, and rewrite default values for existing options.

		Parameters:
			parser          -- original option parser
			is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

		Returns:
			the modified parser.
		"""
		return parser

	@abstractmethod
	def __len__(self):
		"""Return the total number of images in the dataset."""
		return 0

	@abstractmethod
	def __getitem__(self, index):
		"""Return a data point and its metadata information.

		Parameters:
			index - - a random integer for data indexing

		Returns:
			a dictionary of data with their names. It ususally contains the data itself and its metadata information.
		"""
		pass


def get_params(opt, size):
	"""Method to resize or rescale image size"""
	# Get width and height 
	w, h, c = size
	new_h = h
	new_w = w
	if opt.preprocess == 'resize_and_crop':
		new_h = new_w = opt.load_size
	elif opt.preprocess == 'scale_width_and_crop':
		new_w = opt.load_size
		new_h = opt.load_size * h // w
    # Get cut positions
	x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
	y = random.randint(0, np.maximum(0, new_h - opt.crop_size))
    # Define rotation position
	flip = random.random() > 0.5

	return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
	"""Method to apply transformations in images, as convert image to grayscale, resize,  crop and flip """
	transform_list = []
	# Convert image to grayscale 
	if grayscale:
		transform_list.append(transforms.Grayscale(1))
	# Resize image 
	if 'resize' in opt.preprocess:
		osize = [opt.load_size, opt.load_size]
		transform_list.append(transforms.Resize(osize, method))
	# Rescale  image width
	elif 'scale_width' in opt.preprocess:
		transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))
	# Apply crop function in an image
	if 'crop' in opt.preprocess:
		if params is None:
			# Crop image randomly 
			transform_list.append(transforms.RandomCrop(opt.crop_size))
		else:
			# Crop image with the position defined in get_params function
			transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))
	# Apply flip function in an image
	if opt.flip:
		if params is None:
			# Flip image randomly
			transform_list.append(transforms.RandomHorizontalFlip())
		elif params['flip']:
			# Flip image with the position defined in get_params function
			transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
	if convert:
		# Convert images to tensors, in the range [0, 255]
		transform_list += [transforms.ToTensor()]
		# Normalize images with mean and standard devitation
		if grayscale:
			transform_list += [transforms.Normalize((0.5,), (0.5,))]
		else:
			transform_list += [transforms.Normalize(mean = [0.485, 0.456, 0.406],
						 std = [0.229, 0.224, 0.225])]
	return transforms.Compose(transform_list)


# def __none(img):
#   return img

def __make_power_2(img, base, method=Image.BICUBIC):
	""" Method to verify image size is multiple of the the base"""
	ow, oh = img.size
	h = int(round(oh / base) * base)
	w = int(round(ow / base) * base)
   
	if h == oh and w == ow:
		return img
		
	__print_size_warning(ow, oh, w, h)
	return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
	""""Method to rescale image width"""
	ow, oh = img.size
     # Checkig image size
	if ow == target_size and oh >= crop_size:
		# Verify if image is multiple of 4
		return __make_power_2(img, base = 4, method = method)
	w = target_size
	h = int(max(target_size * oh / ow, crop_size))
	# Resize image
	img = img.resize((w, h), method)
	# Verify if image is multiple of 4
	img =__make_power_2(img, base = 4, method = method)
   
	return img

def __padding(img):
	"""Method to apply padding in images"""
	ow, oh = img.size
	# Verify if images are multiple of 4
	h = (math.ceil(oh / 4) * 4)
	w = (math.ceil(ow / 4) * 4) 
	
	if h == oh and w == ow:
		return img
	# Warning information about the size
	__print_size_warning(ow, oh, w, h)
	# Create a new image
	new_img = Image.new("RGB", (w, h))
	# Paste the padding 
	new_img.paste(img, (0, 0))
	return new_img


def __crop(img, pos, size):
	"""Method to crop image"""
	ow, oh = img.size
	x1, y1 = pos
	tw = th = size
	if (ow > tw or oh > th):
		return img.crop((x1, y1, x1 + tw, y1 + th))
	return img


def __flip(img, flip):
	"""Method to flip image"""
	if flip:
		return img.transpose(Image.FLIP_LEFT_RIGHT)
	return img


def __print_size_warning(ow, oh, w, h):
	"""Print warning information about image size(only print once)"""
	if not hasattr(__print_size_warning, 'has_printed'):
		print("The image size needs to be a multiple of 4. "
			  "The loaded image size was (%d, %d), so it was adjusted to "
			  "(%d, %d). This adjustment will be done to all images "
			  "whose sizes are not multiples of 4" % (ow, oh, w, h))
		__print_size_warning.has_printed = True
