"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import collections
import torch.utils.data as data

from PIL import Image
import os
import os.path
import json
from natsort import natsorted


IMG_EXTENSIONS = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
	'.tif', '.TIF', '.tiff', '.TIFF',
]

import scipy.io

from scipy import signal
from skimage.transform import pyramid_gaussian
from scipy.fft import fft, ifft, fftshift, ifftshift
from cv2 import cv2
import numpy as np

def butter_bandpass(low, high, fs, order=5):
	"""Process signal for butter bandpass"""
	nyq = 0.5 * fs
	low_cutoff = low / nyq
	high_cutoff = high / nyq
	# Apply butterworth digital filter 
	b, a = signal.butter(order, [low_cutoff, high_cutoff], btype='band', analog=False)
	return b, a

def butter_bandpass_filter(data, low, high, fs, order=5):
	"""Method to apply butter_bandpass filter"""
	b, a = butter_bandpass(low, high, fs, order=order)
	# Filter a data sequence
	y = signal.lfilter(b, a, data)
	return y

def post_process_image(image):
	"""Post process the image to create a full contrast stretch of the image
	takes as input:
	image: the image obtained from the inverse fourier transform
	return an image with full contrast stretch
	-----------------------------------------------------
	1. Full contrast stretch (fsimage)
	2. take negative (255 - fsimage)
	"""
	a = 0
	b = 255
	c = np.min(image)
	d = np.max(image)
	rows, columns = np.shape(image)
	image1 = np.zeros((rows, columns), dtype=float)
	for i in range(rows):
		for j in range(columns):
			if (d-c) == 0:
				image1[i, j] = ((b - a) / 0.000001) * (image[i, j] - c) + a
			else:
				image1[i, j] = ((b - a) / (d - c)) * (image[i, j] - c) + a

	return np.float32(image1)

#  I = The input grey scale image
#  d0 = Lower cut off frequency
#  d1 = Higher cut off frequency
#  n = order of the filter
def butterworthbpf(I, d0, d1):
	"""Method to apply butterworth digital filter"""
	f = I.astype(np.double)
	ny, nx = f.shape
	new_shape = [1, nx]
	fftI = np.zeros((ny, nx), dtype = np.cdouble)

	for i in range(ny):
		# Apply Fourier transform in input data, also apply Fast Fourier transformation to shift the lower and upper half of vector
		fftI[i] =  fftshift(fft(f[i]))

	filter1 = np.ones(new_shape)
	filter2 = np.ones(new_shape)
	filter3 = np.ones(new_shape)
	
	for j in range(nx):

		dist = abs((j - (nx / 2)))
		# Filling a matrix with 0 or 1, dependent on lower and higher frequency size
		filter1[0, j] = 1 if dist <= d1 else 0 # 1 / (1 + (dist / d1) ** (2 * n))
		filter2[0, j] = 1 if dist <= d0 else 0 # 1 / (1 + (dist / d0) ** (2 * n))
		filter3[0, j] = 1.0 - filter2[0, j]
		filter3[0, j] = filter1[0, j] * filter3[0, j]

	filtered = np.zeros((ny, nx), dtype = np.cdouble)

	for i in range(ny):
		filtered[i] = filter3[0] * fftI[i]

	ifftI = np.zeros((ny, nx), dtype = np.cdouble)

	for i in range(ny):
		# Calculate inverse discret Fourier transform from filtered signal  
		ifftI[i] = ifft(ifftshift(filtered[i]))

	return post_process_image(ifftI.real)


def get_hr_array(path):
	"""Method to load heart rate path"""
	mat = scipy.io.loadmat(path)
	return mat['bpm'][0] 

def make_feature_image_dataset(json_path, Pl, Fps, low_cut, high_cut, out_dir, opt):
	"""Method to create feature image"""

	# This method combines one or more path names into a single path
	dataset_info_json_path = os.path.join(out_dir, "dataset_info.json")
	# To check whether the specified path is an existing directory or not
	if os.path.isdir(out_dir):
		# If specified path exist, open it 
		f = open(dataset_info_json_path)
		# Return json object  
		return json.load(f)	
	f = open(json_path)
	# Load json achieve
	json_file = json.load(f)
	make_dir(out_dir)
	feature_image_json = {}
	print('opt.video_fps:', opt.video_fps)
	dataset_json = json_file
	keys = list(json_file.keys())

	if opt.video_fps != 30: # redução do fps do video para 30 diretamente no json # Funciona apenas para fps multiplos de 30
		print(f"Reduce {opt.video_fps} to 30")
		for key in keys:
			#print(len(self.dataset_json[key]['bottom_face']))
			for i in range(len(dataset_json[key]['bottom_face']), 0, -1):
				if ((i-1) % (opt.video_fps/30)) != 0:
					del dataset_json[key]['bottom_face'][i - 1]
					del dataset_json[key]['right_eye'][i - 1]
					del dataset_json[key]['left_eye'][i - 1]
	
	# Create a loop for load subjects and keys from json file 
	for name_subject, subject in json_file.items():
		C = []
		subject = collections.OrderedDict(natsorted(subject.items()))
		feature_image_json[name_subject] = {"PPG" : '', "feature_image" : []}

		if 'PPG' in subject.keys():
			# Create a new dictionary without PPG
			feature_image_json[name_subject]['PPG'] = subject.pop("PPG")
         # To check subject's keys and make a list with it 
		landmarks_names = list(subject.keys())
		if 'middle_face' in landmarks_names:
			landmarks_names.remove('middle_face')
		print(f'Face landmaks used: {landmarks_names}')
		# Counting the number of frames in facial landmarks 
		length = len(subject[landmarks_names[0]])
		dir_subject = os.path.join(out_dir, name_subject)
		# Create a new directory for each subject 
		make_dir(dir_subject)
		Set = []
		for i in range(length):
			frames = []
			for name in landmarks_names:
				# Read every frames in all positions
				frame = cv2.imread(subject[name][i])
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				#  Reducing images by a constant scale factor
				pg = tuple(pyramid_gaussian(frame, multichannel = True, preserve_range = True)) 
				desired_level = pg[Pl]
				w, h, c = desired_level.shape
				C0 = np.array(desired_level).reshape((w * h, c))
				frames.append(C0)
			frames = np.concatenate(frames, axis = 0)
			C.append(frames)
			if len(C) == 30:
				C = cv2.rotate(np.array(C), cv2.ROTATE_90_CLOCKWISE)
				Set.append(C)
				C = []
		
		target_gt_diff = 0
		if opt.phase != 'test':
			# Get the difference between the number of sets of frames and the amount of BPM in the mat file.
			target_gt_diff = len(Set) - len(get_hr_array(feature_image_json[name_subject]["PPG"]))
		 
		if target_gt_diff > 0: # If there are more frames than the amount of BPM they are discarded
			print(f"Difference in value between number of featured images and BPMs of {name_subject}: {target_gt_diff}")
			Set = Set[:-target_gt_diff] 
			print(f"Discarded extra features images, new width of the list: {len(Set)}")
		 
		for index, cols in enumerate(Set):
			N = []
			for i in range(3):
				channel = np.array(cols)[:, :, i]
				# Apply butterworth filter to  return a frequency response 
				N.append(butterworthbpf(channel, low_cut, high_cut))
			K = cv2.merge(np.array(N))
			out_path = os.path.join(dir_subject, f"{index}.npy")
			# Store subject feature image 
			feature_image_json[name_subject]["feature_image"].append(out_path)
			np.save(out_path, K)
	
	with open(dataset_info_json_path, 'w') as outfile:
		json.dump(feature_image_json, outfile, indent = 1)
	return feature_image_json

def make_dir(path):
	"""Check if the directory exists otherwise create one."""
	if not os.path.isdir(path):
		os.mkdir(path)

def is_image_file(filename):
	"""Check if the file extension belongs to a image group"""
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_json_dataset(dir, max_dataset_size=float("inf")):
	"""Accepts file object, parses the JSON data, populates a Python dictionary with the data and returns it """
	return json.load(open(dir))

def make_dataset(dir, max_dataset_size=float("inf")):
	"""Checks if the directory is valid and returns a list with the path of sorted images"""
	images = []
	assert os.path.isdir(dir), '%s is not a valid directory' % dir
	
	for root, _, fnames in natsorted(os.walk(dir)):
		for fname in fnames:
			if is_image_file(fname):
				path = os.path.join(root, fname)
				images.append(path)
							
	return natsorted(images[:min(max_dataset_size, len(images))])


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