"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
	-- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
	-- <__len__>:                       return the size of dataset.
	-- <__getitem__>:                   get a data point from data loader.
	-- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from .base_dataset import BaseDataset
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from typing import Iterator, Sequence
from .image_folder import make_feature_image_dataset, make_json_dataset
import numpy as np
import random as rd
import os

class SubsetSequentialSampler(Sampler):
	r"""Samples elements sequentially from a given list of indices, without replacement.
	Args:
		indices (sequence): a sequence of indices
		generator (Generator): Generator used in sampling.
	"""
	indices: Sequence

	def __init__(self, indices: Sequence, generator=None) -> None:
		self.indices = indices
		self.generator = generator

	def __iter__(self) -> Iterator:
		return (self.indices[i] for i in range(len(self.indices)))

	def __len__(self) -> int:
		return len(self.indices)

def find_dataset_using_name(dataset_name):
	"""Import the module "data/[dataset_name]_dataset.py".

	In the file, the class called DatasetNameDataset() will
	be instantiated. It has to be a subclass of BaseDataset,
	and it is case-insensitive.
	"""
	dataset_filename = "data." + dataset_name + "_dataset"
	try:
		datasetlib = importlib.import_module(dataset_filename)
	except:
		datasetlib = importlib.import_module('EVM.' + dataset_filename)

	dataset = None
	target_dataset_name = dataset_name.replace('_', '') + 'dataset'
	for name, cls in datasetlib.__dict__.items():
		if name.lower() == target_dataset_name.lower() \
		   and issubclass(cls, BaseDataset):
			dataset = cls

	if dataset is None:
		raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

	return dataset

def get_option_setter(dataset_name):
	"""Return the static method <modify_commandline_options> of the dataset class."""
	dataset_class = find_dataset_using_name(dataset_name)
	return dataset_class.modify_commandline_options
																							
def get_indices(dataset_json, indices):  
	"""Return indexes e keys of feature image from dataset json"""                                                   
	new_indices = []
	for key in indices:
		for index, data in enumerate(dataset_json[key]['feature_image']):
			new_indices.append((key, index))
	return new_indices

def create_dataset(opt):
	"""Create a dataset given the option.

	This function wraps the class CustomDatasetDataLoader.
		This is the main interface between this package and 'train.py'/'test.py'

	Example:
		>>> from data import create_dataset
		>>> dataset = create_dataset(opt)
	"""
	
	if opt.model in ("evm_cnn"):
		# "../RGB_dataset_ideal_filter_UBFC_magnified" # ../RGB_dataset_ideal_filter_float_bottom_face"
		feature_image_dataset_dir = os.path.join(opt.results_dir, f"feature_{opt.feature_image_path.split(os.sep)[-2]}")
		dataset_json = make_feature_image_dataset(opt.feature_image_path, 6, opt.video_fps, 0.75, 4, feature_image_dataset_dir, opt)
		opt.feature_image_path = feature_image_dataset_dir
	else:
		dataset_json = make_json_dataset(opt.feature_image_path, opt.max_dataset_size)
	
	dataset_size = len(dataset_json)
	indices = list(dataset_json.keys())

	train_val_split = int(np.round(0.9 * dataset_size))
	test_split = train_val_split + int(np.round(0.1 * dataset_size))
	
	train_and_val_indices, test_indices =  indices[:train_val_split], indices[train_val_split:test_split]   
	
	print(f"Number of videos for train & val, and for test: {train_and_val_indices, test_indices}")

	train_indices = get_indices(dataset_json, train_and_val_indices)
	
	if opt.do_not_split_for_test:
		rd.shuffle(train_indices)
	
	val_split = int(np.round(0.75 * len(train_indices)))
	train_indices, val_indices = train_indices[:val_split], train_indices[val_split:]
	test_indices = get_indices(dataset_json, test_indices)
	
	print(f"Videos indexes for train, val and test: {[len(train_indices), len(val_indices), len(test_indices)]}")
	
	if opt.phase in ("train"):
	   
		if not opt.do_not_split_for_test:
			val_indices = val_indices + test_indices
			test_indices = []
	  
		print(f"Frames for train: {[len(train_indices)]}")
		print(f"Frames for val: {[len(val_indices)]}")
		print(f"Frames for test: {[len(test_indices)]}")
		
		train_sampler = SubsetRandomSampler(train_indices)
		valid_sampler = SubsetRandomSampler(val_indices)

		train_data_loader = CustomDatasetDataLoader(opt, train_sampler, dataset_json, len(train_indices))
		val_data_loader = CustomDatasetDataLoader(opt, valid_sampler, dataset_json, len(val_indices))

		train_dataset = train_data_loader.load_data()
		val_dataset = val_data_loader.load_data()
		return [train_dataset, val_dataset]
	else:
		test_indices = test_indices if opt.do_not_split_for_test else train_indices + val_indices + test_indices
		print(f"Frames for test: {[len(test_indices)]}")
		
		test_sampler = SubsetSequentialSampler(test_indices)
		test_data_loader = CustomDatasetDataLoader(opt, test_sampler, dataset_json, len(test_indices))
		test_dataset = test_data_loader.load_data()
		return test_dataset

class CustomDatasetDataLoader():
	"""Wrapper class of Dataset class that performs multi-threaded data loading"""

	def __init__(self, opt, sampler, dataset_list, length):
		"""Initialize this class

		Step 1: create a dataset instance given the name [dataset_mode]
		Step 2: create a multi-threaded data loader.
		"""
		self.opt = opt
		dataset_class = find_dataset_using_name(opt.dataset_mode)
		self.dataset = dataset_class(opt, dataset_list, length)
		print("dataset [%s] was created" % type(self.dataset).__name__)
		self.dataloader = torch.utils.data.DataLoader(
			self.dataset,
			batch_size = opt.batch_size,
			num_workers = int(opt.num_threads),
			sampler = sampler
		)

	def load_data(self):
		return self

	def __len__(self):
		"""Return the number of data in the dataset"""
		return min(len(self.dataset), self.opt.max_dataset_size)

	def __iter__(self):
		"""Return a batch of data"""
		for i, data in enumerate(self.dataloader):
			if i * self.opt.batch_size >= self.opt.max_dataset_size:
				break
			yield data
