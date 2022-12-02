from __future__ import print_function
import torch
# import pickle
import numpy as np

from .data_utils import butter_bandpass_filter

from natsort import natsorted
import json
import scipy.io
from cv2 import cv2
import math

class BaselineDataset():
	"""Preprocessing class of Dataset class that performs multi-threaded data loading

	"""
	def __init__(self, opt, phase):
		"""Initialize this dataset class.

		Parameters:
			opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

			The self.dataset is a list of facial data, the length of the list is 18, and each element is a torch tensor of shape [2852, 3, 64, 64]
			The self.maskset is the corresponding mask data, constructed of 0 and 255, so it determines the landmarks we're using in self.dataset  

		"""
		# get the image directory

		self.phase = phase
		self.opt = opt
		
		with open(opt.feature_image_path, 'r') as fp:
			json_file = json.load(fp)
		
		arb_list = []
		
		support_json = {}
		query_json = {}
				  
		self.dataset_json = json_file
		self.keys = list(self.dataset_json.keys())

		if self.opt.video_fps != 30: # Reduction of video fps to 30 # Works only for multiple fps of 30
			print(f"Reduce {self.opt.video_fps} to 30")
			for key in self.keys:
				PPG_len = scipy.io.loadmat(self.dataset_json[key]['PPG'])['pulseOxRecord'].size
				for i in range(len(self.dataset_json[key]['bottom_face']),0,-1):
					if ((i-1) % (self.opt.video_fps/30)) != 0:
						del self.dataset_json[key]['bottom_face'][i - 1]
						del self.dataset_json[key]['right_eye'][i - 1]
						del self.dataset_json[key]['left_eye'][i - 1]
				if len(self.dataset_json[key]['bottom_face']) > PPG_len:
					del self.dataset_json[key]['bottom_face'][PPG_len:]
					del self.dataset_json[key]['right_eye'][PPG_len:]
					del self.dataset_json[key]['left_eye'][PPG_len:]

		if not self.opt.is_raw_dataset: # Create a list with the number of frames for all subjects
			for key in self.keys:
				arb_list.append(len(self.dataset_json[key]['image']))
		else:
			for key in self.keys:
				arb_list.append(len(self.dataset_json[key]['bottom_face']))
		
		arb_val = min(arb_list) # Select the smallest number of frames
		""" Separate the amount of frames for pre-training, query and support """
		pretrain_val = round(arb_val*0.2)
		if pretrain_val < 420:
			pretrain_val = 420
		supp_val = round(pretrain_val*0.45)
		query_val = pretrain_val - supp_val

		sub_val = arb_val - (math.floor(arb_val/pretrain_val)*pretrain_val) # Difference to  arb_val be a multiple of pretrain_val

		if self.phase != 'test' and self.opt.is_raw_dataset: # Load mat file id phase is test and opt is is_raw_dataset
			for subj_name in self.keys:
				self.dataset_json[subj_name]['PPG'] = \
					scipy.io.loadmat(self.dataset_json[subj_name]['PPG'])['pulseOxRecord'][0].tolist()

		train_split = 0
		val_split = 0
  
		""" Split subjects for train, val and test(eval) """
		if opt.do_not_split_for_test: 
			train_val_split = int(np.math.floor(0.88 * len(self.keys)))
			train_split = int(np.math.floor(0.95 * train_val_split))
			val_split = train_val_split
			test_split = len(self.keys) - train_val_split
		else:
			test_split = len(self.keys)
		
		self.dataset_json_sup_query = dict(natsorted(list(self.dataset_json.items()))[:train_split]) if not self.phase in ("test", "eval") else self.dataset_json
		
		print(f"Train split: {train_split}")
		print(f"Val split: {val_split}")
		print(f"Test split: {test_split}")
		
		i = 0
		if not self.phase in ("test", "eval"):
			for subname, sub in self.dataset_json_sup_query.items():
				for j in range(0, arb_val - sub_val, pretrain_val): 
					support_json[i] = {'image': [], 'mask': [], 'ppg': []}
					query_json[i] = {'image': [], 'mask': [], 'ppg': []}
					for key, val in sub.items():
						support_json[i][key] = val[j : j + supp_val] if self.phase != 'pretrain' else val[j : j + pretrain_val]
						query_json[i][key] = val[j + supp_val : j + pretrain_val]
					i += 1
		
		if self.phase == 'pretrain':
			self.dataset_json = support_json 
			self.keys = list(self.dataset_json.keys())
			self.num_tasks = len(self.keys)
			self.task_len = [pretrain_val for i in range(len(self.keys))]
		elif self.phase == 'supp':
			self.dataset_json = support_json 
			self.keys = list(self.dataset_json.keys())
			self.num_tasks = len(self.keys)
			self.task_len = [supp_val for i in range(len(self.keys))]       
		elif self.phase == 'query':
			self.dataset_json = query_json
			self.keys = list(self.dataset_json.keys())
			self.num_tasks = len(self.keys)
			self.task_len = query_val
		elif self.phase == 'val':
			self.dataset_json = dict(natsorted(list(self.dataset_json.items()))[train_split:val_split])
			self.keys = list(self.dataset_json.keys())
			self.num_tasks = len(self.keys)
			self.task_len = arb_val
		elif self.phase in ("test", "eval"):
			self.dataset_json = dict(natsorted(list(self.dataset_json.items()))[-test_split:])
			self.keys = list(self.dataset_json.keys())
			self.num_tasks = len(self.keys)
			if self.opt.is_raw_dataset:
				self.task_len = {key: len(self.dataset_json[key]['bottom_face']) for key in self.keys}
			else:
				self.task_len = {key: len(self.dataset_json[key]['image']) for key in self.keys}

		print(f'self.keys: {self.keys}')

		self.length = 0
		for i in range(len(self.keys)):
		  self.length += arb_val - self.opt.win_size
	 
	
	def __getframe__(self, subject, i):
		"""Return the facial landmarks in a single image.

		Parameters:
			subject: the actual subject in the json file.
			i: Number of frames the subject has.

		Returns the Float and Byte Tensor of the frame and mask respectively 
		"""
		if self.opt.is_raw_dataset:
			bottom_face = cv2.imread(subject['bottom_face'][i])
			bottom_face = cv2.resize(bottom_face, (0,0), fx=0.5, fy=0.5)
			left_eye = cv2.imread(subject['left_eye'][i])
			right_eye = cv2.imread(subject['right_eye'][i])
			eyes = cv2.hconcat([right_eye, left_eye])
			bottom_face = cv2.vconcat([eyes, bottom_face])
			bottom_face = bottom_face[:-100, :]
			h, w, c = bottom_face.shape
			frame = cv2.resize(bottom_face, (0,0), fx=0.32, fy=0.32)
			frame = np.transpose(frame, (2, 0, 1))
			mask = np.ones_like(frame) * 255
			return torch.FloatTensor(frame), torch.ByteTensor(mask)
		else:
			frame = cv2.imread(subject['image'][i])
			mask = cv2.imread(subject['mask'][i])
			frame = np.transpose(frame, (2, 0, 1))
			mask = np.transpose(mask, (2, 0, 1))
			return torch.FloatTensor(frame), torch.ByteTensor(mask)
		 
	def __getitem__(self, items):
		"""Return a data point and its metadata information.

		Parameters:
			items -- [task_number, index of data for specified task]
			items[0] -- a integer in range 0 to 4 in train mode, only 0 available in test mode
			items[1] -- determined by the length of the video

		Returns a dictionary that contains input, PPG, diff and orig
			input - - a set of frames from the pickle file (60 x 3 x 64 x 64)
			PPG - - the corresponding signal (60)
		"""

		inputs = []
		masks = []
		subject = self.dataset_json[items[0]]
		
		for i in range(items[1], items[1] + self.opt.win_size):
			frame, mask = self.__getframe__(subject, i)
			inputs.append(frame.clone())
			masks.append(mask.clone())
		  
		if self.phase in ('test'):
			ppg = 0 # phase 'test' does not have ppg
		else:
			if not self.opt.is_raw_dataset:
				ppg = torch.FloatTensor(subject['ppg'][items[1] : items[1] + self.opt.win_size]).clone()
			else:
				ppg = torch.FloatTensor(subject['PPG'][items[1] : items[1] + self.opt.win_size]).clone()
			if len(ppg) != 60: # check if ppg arrys are size 60
				print(f"Subject: {subject['left_eye'][0]} -> len(ppg): {len(ppg)}")
			ppg = self.quantify(ppg) 

		inputs = np.stack(inputs)
		inputs = torch.from_numpy(inputs)
		masks = np.stack(masks)
		masks = torch.from_numpy(masks)

		self.baseline_procress(inputs, masks.clone())

		return {'input': inputs, 'PPG': ppg}

	def __len__(self):
		"""Return the total number of images in the dataset."""

		return self.length

	def quantify(self, rppg):
		"""Quantify arry of current ppg
      
      	Parameters:
        	rppg: ppg batch corresponding to 1 second 
      	"""
		quantified = torch.empty(rppg.shape[0], dtype=torch.long)
		tmax = rppg.max()
		tmin = rppg.min()
		interval = (tmax - tmin)/39
		for i in range(len(quantified)):
			quantified[i] = ((rppg[i] - tmin)/interval).round().long()
		return quantified
	
	def baseline_procress(self, data, mask):
		"""Baseline procress
      
      	Parameters:
        	data: a set of frames from the pickle file (60 x 3 x 64 x 64)
			mask: mask from the frames from the pickle file
      	"""
		mask /= 255
		mask = mask.float()

		# pdb.set_trace()
		input_mean = data.sum(dim=(0, 2, 3), keepdim=False) / \
			mask.sum(dim=(0, 2, 3), keepdim=False)  # mean of W H T
		for i in range(data.shape[1]):
			data[:, i, :, :] = data[:, i, :, :] - input_mean[i]  # minus the total mean
		data = data*mask
		
		x_hat = data.sum(dim=(2, 3), keepdim=False)/ \
			mask.sum(dim=(2, 3), keepdim=False)  # mean of H T
		G_x = np.empty(x_hat.size())  # filtered x_hat

		for i in range(data.shape[1]):  # shape 1 is RGB channels
			# pdb.set_trace()
			G_x[:, i] = butter_bandpass_filter(x_hat[:, i], 1, 8, 30, order=3)
			for j in range(data.shape[0]):
				data[j, i, :, :] = data[j, i, :, :] - \
						(x_hat[j, i] - G_x[j, i])
		data = data*mask
		# pdb.set_trace()
		return data

	def __call__(self, idx):
		inputs = []
		masks = []
		items = [idx, 0]

		if not self.isTrain:
			# pdb.set_trace()
			# decision = 0
			new_index = items[1] % (
				self.task_len - (self.opt.batch_size + self.opt.fewshots)*self.opt.win_size)
			for i in range(new_index, new_index + 15*self.opt.win_size):
				frame = self.dataset[items[0]][i].clone()
				mask = self.maskset[items[0]][i].clone()
				inputs.append(frame)
				masks.append(mask)
			ppg = self.ppg_dataset[items[0]][new_index: new_index + 15*self.opt.win_size].clone()
			# orig = self.original[items[0]][new_index: new_index + 15*self.opt.win_size].clone()
		else:
			for i in range(items[1], items[1] + 15*self.opt.win_size):
				frame = self.dataset[items[0]][i].clone()
				mask = self.maskset[items[0]][i].clone()
				inputs.append(frame)
				masks.append(mask)
			ppg = self.ppg_dataset[items[0]][items[1]: items[1] + 15*self.opt.win_size].clone()
			# orig = self.original[items[0]][items[1]: items[1] + 15*self.opt.win_size].clone()

		inputs = np.stack(inputs)
		inputs = torch.from_numpy(inputs)
		masks = np.stack(masks)
		masks = torch.from_numpy(masks)

		self.baseline_procress(inputs, masks.clone())
		ppg = self.quantify(ppg)

		return {'input': inputs, 'PPG': ppg}
