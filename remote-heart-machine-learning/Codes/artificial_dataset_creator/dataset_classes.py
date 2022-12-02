from dataset_analyzer.simulate import ppg_simulate
from utils import check_video_pattern, print_log, make_dir
from dataset_analyzer.ppg_analyzer import array_flatten, ppg_to_bpm
from xml.etree import ElementTree as ET
from scipy.io import loadmat, savemat
from operator import itemgetter
import numpy as np
import pyedflib
import datetime
import logging
import heartpy
import natsort
import abc
import cv2
import os

class Dataset(metaclass=abc.ABCMeta):
	"""Abstract super class responsible for defining the required methods that must be implemented to use a dataset."""
	def __init__(self):
		self.paths = []
		self.image_extension = ''

	@abc.abstractmethod
	def map_dataset(self):
		"""Function that traverses a folder structure to map raw dataset videos.

		Return
		------
		paths : Iterable
			List of dictionaries containing the following structure: \n
			>>> [
			>>> 	{
			>>> 		'subject_folder_name': 'Subject name or his folder', 
			>>> 		'subdir_folder_name': "Name of the internal folder containing the subject's video", 
			>>> 		'video_path': 'Relative or absolute path of the video with filename', 
			>>> 		'ppg_path': 'Relative or absolute path of the PPG with filename'
			>>> 	},
			>>> 	...
			>>> ]
		"""
		raise NotImplementedError

	@abc.abstractmethod
	def ppg_flatten(self, pulse_path, out_path):
		"""Function that opens the PPG file, retrieves its values, applies any processing and saves it in the output path.

		Keyword arguments
		-----------------
		pulse_path : str
			PPG file path, it contains the filename.
		
		out_path : str
			Output path that will be used to save the processed PPG.
		"""
		raise NotImplementedError

	@abc.abstractmethod
	def save_video(self, frame: np.ndarray, out_path: str, currentframe: int=None):
		"""Function that saves a frame to disk.

		Notes
		-----
		- On disk, the frame can be an image file that has its position in the name, e.g.: 

		>>> cv2.imwrite(os.path.join(out_path, 'Frame%05d.pgm' % currentframe), frame)
		
		- Also, the frame can be inside a video file, e.g.:\n
			
		>>> out = self.writers.get(out_path, None)
		>>> if out is None or isinstance(out, cv2.VideoWriter) == False:\n
		>>> 	out[out_path] = cv2.VideoWriter(\n
		>>> 		os.path.join(out_path, f'{self.name}.avi'), \n
		>>> 		cv2.VideoWriter_fourcc(*'DIVX'), \n
		>>> 		30, \n
		>>> 		(self.ROI_size, self.ROI_size)\n
		>>> 	)\n

		>>> out.write(frame)\n
		>>> self.writers[out_path] = out

		Keyword arguments
		-----------------
		frame : ndarray
			Image that will be written to disk.
		
		out_path : str
			Path of the output folder where the frame will be written.
		
		current_frame : int
			Number indicating the position of the frame in the video (starts at zero)
		"""
		raise NotImplementedError
	
	def demosaic_video(self, video_path: str, out_path: str, cv2_cvt_code=cv2.COLOR_BAYER_BG2BGR, video_name_pattern='Frame%05d.pgm') -> None:
		"""Function that receives the folder structure to apply demosaic to a video.

		Keyword arguments
		-----------------
		dataset : Dataset
			The `dataset` object must belong to a class that enherit the class `Dataset` and must have implemented all its methods.
			It must have the following variable:
				dataset.image_extension : str
					Filename extension that the frame will have on disk (without the dot before the extension)
		
		video_path : str
			Video path that must be processed. It must be a sequence of images within a folder. Each frame will be readed with the \
				`cv2.imread` method. E.g.:
			>>> '/path/to/video_folder/Frame%05d.png' 
			# or just: 
			>>> '/path/to/video_folder'
		
		out_path : str
			Path of the output folder where the frame will be written. It will be written with the same name as it was read.
		
		video_name_pattern : str
			The variable `video_name_pattern` must be a string with the following formatter: (%d). E.g.:
				- 'Frame%05d.pgm' which will be done: pattern % 0 (%05d says the sequence will be five digits \
					long, with the first ones filled with zeros: Frame00000.pgm, Frame00001.pgm, Frame00002.pgm, ...).
		"""
		assert os.path.isdir(out_path), "The `out_path` param must be a directory"

		start = video_name_pattern.split('%')[0]
		end = video_name_pattern.split('d')[-1]

		# create a filter base on the `video_name_pattern`
		def frame_filter(filename: str):
			middle = filename.replace(start, '').replace(end, '')
			return filename.startswith(start) and filename.endswith(end) and middle.isdigit()

		if os.path.isdir(video_path) == False:
			# get parent folder
			video_dir = os.sep.join(video_path.split(os.sep)[:-1])
		else:
			video_dir = video_path

		# filter the folder by `video_name_pattern` and sort it
		frame_names = natsort.natsorted(filter(frame_filter, os.listdir(video_dir)))

		# for each frame that satisfies the requirement
		for frame_name in frame_names:
			frame_path = os.path.join(video_dir, frame_name)

			# based on: https://gist.github.com/bbattista/8358ccafecf927ae1c58c944ab470ffb
			# but the same process could be done with `cv2.demosaicing`: https://answers.opencv.org/question/236504/pixel-order-for-demosaicing/
			frame = np.right_shift(cv2.imread(frame_path, cv2.IMREAD_UNCHANGED), 8)

			# Perform a Bayer Reconstruction
			frame = cv2.cvtColor(frame, cv2_cvt_code).astype('uint8')

			# write it with the same name as it was read
			cv2.imwrite(os.path.join(out_path, frame_name.replace(end, f'.{self.image_extension}')), frame)

	def demosaic_RGGB_dataset(self, out_dataset_path: str, cv2_cvt_code=cv2.COLOR_BAYER_BG2BGR, video_name_pattern='Frame%05d.pgm') -> None:
		"""Function that receives the folder structure to process each video with the `demosaic_video` funtion.
		
		Notes
		-----
		- All paths will be checked during the process. An error will be issued without harming other procedures.

		Keyword arguments
		-----------------
		dataset : Dataset
			The `dataset` object must belong to a class that enherit the class `Dataset` and must have implemented \
				the `save_video` and `ppg_flatten` methods.
			It must execute `dataset.map_dataset` before calling this function to get the following variable:
				dataset.paths: Iterable
					List of dictionaries that must have the following struture:\n
					>>> [
					>>> 	{
					>>> 		'subject_folder_name': 'Subject name or his folder', 
					>>> 		'subdir_folder_name': "Name of the internal folder containing the subject's video", 
					>>> 		'video_path': 'Relative or absolute path of the video with filename', 
					>>> 		'ppg_path': 'Relative or absolute path of the PPG with filename'
					>>> 	},
					>>> 	...
					>>> ]
		
		out_dataset_path : str
			Path of the output folder where you want to save the demosaiced dataset. It should not exist to not overwrite the previous dataset.
		
		cv2_cvt_code : int
			Integer that the `cv2` module uses with the `cv2.cvtColor` function. 
			The following issue is worth reading: https://github.com/opencv/opencv/issues/19629

		video_name_pattern : str
			The `video_name_pattern` parameter must be a string with the following formatter or none: (%d). E.g.:
				- 'Frame%05d.pgm' which will be done: pattern % 0 (%05d says the sequence will be five digits \
					long, with the first ones filled with zeros: Frame00000.pgm, Frame00001.pgm, Frame00002.pgm, ...). 
		"""
		start_time = datetime.datetime.now()
		print_log(f"\nStart of execution: {start_time}", None)
		assert issubclass(self.__class__, Dataset), "The `dataset` object must belong to a class that enherit the class `Dataset`"

		exceptions_count = 0

		# for each video inside the mapped dataset
		for path in self.paths:
			subject_folder_name = path['subject_folder_name']
			video_path = path['video_path']

			out_path = os.path.join(out_dataset_path, subject_folder_name, 'RGB_demosaiced')
			make_dir(out_path)
			
			try:
				assert check_video_pattern(video_path, video_name_pattern, None), f"The following `video_path` wasn't found: {video_path}"
				self.demosaic_video(
					video_path=video_path,
					out_path=out_path, 
					cv2_cvt_code=cv2_cvt_code,
					video_name_pattern=video_name_pattern
				)
			except Exception as e:
				print_log(f"\nUnmapped Exception: \n{logging.traceback.format_exc()}", None)
				exceptions_count += 1

		print_log(f"\nNumber of Exceptions: {exceptions_count}", None)
		end_time = datetime.datetime.now()
		print_log(f"\nEnd of execution: {end_time}", None)

		# it retrives the duration
		duration = end_time - start_time
		# it divides by 3600 and will have (hours, seconds left)
		hours = divmod(duration.total_seconds(), 3600)
		# it divides the seconds left by 60 and will have (minutes, seconds)
		minutes = divmod(hours[1], 60)

		print_log(f"\nTotal runtime: {hours[0]} hours and {minutes[0]} minutes", None)


class MRNirp(Dataset):
	"""This is a `Dataset` subclass. It implements the required methods to use the MR-Nirp indoor dataset."""
	def __init__(self, image_extension: str='pgm'):
		"""
		Keyword arguments
		-----------------
		image_extension : str
			Filename extension that the frame will have on disk (without the dot before the extension)
		"""
		self.paths = []
		assert isinstance(image_extension, str), f"The `image_extension` parameter must be a string: {image_extension}"
		assert image_extension.isalpha(), f"The `image_extension` parameter must only contain letters: {image_extension}"
		self.image_extension = image_extension
	
	def map_dataset(self, base_dataset_path: str, subdir_name=['NIR'], video_name_pattern='Frame%05d.pgm'):
		"""Function that traverses a folder structure to map raw dataset videos. Exclusive to the MR-Nirp indoor dataset.

		Notes
		-----
		- About the subject folder:
			- It must contain a folder named 'PulseOx' with a file named 'pulseOx.mat';
			- It must contain a folder named 'RGB' or 'NIR';
			- The `video_name_pattern` file must exist inside the 'RGB' or 'NIR' folder.

		Keyword arguments
		-----------------
		base_dataset_path : str
			It is the base dataset folder path (dataset base is the one that must be processed). \
				It must exist and contain the following structure:
				- {base_dataset_path}/{sujeito_folder}/{subdir_name}/{video_name_pattern}
				- {base_dataset_path}/{sujeito_folder}/PulseOX/pulseOx.mat
		
		subdir_name : list
			List of folders that are inside (and must be processed) each subject's folder. E.g.:
			- {base_dataset_path}/{sujeito_folder}/RGB/{video_name_pattern} (must have a video inside)
			- {base_dataset_path}/{sujeito_folder}/NIR/{video_name_pattern}
		
		video_name_pattern : str
			The variable `video_name_pattern` must be a string with the following formatter or none: (%d). E.g.:
				- 'Frame%05d.pgm' which will be done: pattern % 0 (%05d says the sequence will be five digits \
					long, with the first ones filled with zeros: Frame00000.pgm, Frame00001.pgm, Frame00002.pgm, ...).

		Return
		------
		paths: Iterable
			List of dictionaries containing the following structure: \n
		>>> [
		>>> 	{
		>>> 		'subject_folder_name': 'Subject name or his folder', 
		>>> 		'subdir_folder_name': "Name of the internal folder containing the subject's video", 
		>>> 		'video_path': 'Relative or absolute path of the video with filename', 
		>>> 		'ppg_path': 'Relative or absolute path of the PPG with filename'
		>>> 	},
		>>> 	...
		>>> ]
		"""
		# START OF VERIFICATION OF FOLDERS AND BASE FILES

		# The `base_dataset_path` folder should exist as a directory.
		assert os.path.isdir(base_dataset_path), f"The `base_dataset_path` parameter is not a directory: {base_dataset_path}"

		# for each folder inside the base dataset
		for subject_folder_name in os.listdir(base_dataset_path):
			subject_folder = os.path.join(base_dataset_path, subject_folder_name)

			# to assume that the `subject_folder` contains a subject we have to check:
			# if it is a directory
			# if it contains a folder named 'PulseOX' and a file with the path '{subject}/PulseOX/pulseOx.mat'
			# if it contains one of the following folders: RGB or NIR
			if os.path.isdir(subject_folder) and \
				os.path.isdir(os.path.join(subject_folder, 'PulseOX')) and \
				os.path.exists(os.path.join(subject_folder, 'PulseOX', 'pulseOx.mat')) and \
				(os.path.isdir(os.path.join(subject_folder, 'RGB')) or \
					os.path.isdir(os.path.join(subject_folder, 'NIR'))):
				# for each folder within the subject's directory
				for subdir_folder_name in os.listdir(subject_folder):
					subdir_folder = os.path.join(subject_folder, subdir_folder_name)

					# it verifies if it is a directory and if it is setted to be used
					if os.path.isdir(subdir_folder) and subdir_folder_name in subdir_name:
						video_path = os.path.join(subdir_folder, video_name_pattern)
						
						# if the name pattern is valid to the respective video:
						if check_video_pattern(video_path, video_name_pattern, None):
							self.paths.append({
								'subject_folder_name': subject_folder_name, 
								'subdir_folder_name': subdir_folder_name, 
								'video_path': video_path,
								'ppg_path': os.path.join(subject_folder, 'PulseOX', 'pulseOx.mat')
							})

		# naturally order the path list
		self.paths = natsort.natsorted(self.paths, key=itemgetter(*['subject_folder_name']))

	def ppg_flatten(self, pulse_path, out_path, ppg_fps) -> None:
		"""Function that opens the MATLAB file, retrives the value of the key named 'pulseOxRecord', flatten the value, \
			overwrites and saves the file as MATLAB in the output path.

		Keyword arguments
		-----------------
		pulse_path : str
			MATLAB file path that contains the PPG.

		out_path : str
			Path of output folder that the handled PPG should be saved as 'pulseOx.mat'.

		ppg_fps : int or real
			Sampling rate that the PPG capture device used.
		"""
		assert os.path.exists(pulse_path), f"The `pulse_path` parameter was not found as a file: {pulse_path}"

		mat = loadmat(pulse_path)

		# Retrives the PPG from the dictionary
		pulse = mat['pulseOxRecord']

		# it removes the multivalued fiels at the PPG using the strategy of flattening
		ppg = array_flatten(pulse)

		bpm = ppg_to_bpm(ppg, ppg_fps=ppg_fps, stride=4, segment_width=4)

		mat['pulseOxRecord'] = ppg
		mat['bpm'] = bpm
		mat['numPulseSample'] = len(ppg)

		savemat(os.path.join(out_path, 'pulseOx.mat'), mat)

	def save_video(self, frame: np.ndarray, out_path: str, currentframe: int=None):
		"""Function that saves a frame to disk.

		Notes
		-----
		- On disk, the frame can be an image file that has its position in the name, e.g.: 

		>>> cv2.imwrite(os.path.join(out_path, 'Frame%05d.pgm' % currentframe), frame)

		- Also, the frame can be inside a video file, e.g.:\n
			
		>>> out = self.writers.get(out_path, None)
		>>> if out is None or isinstance(out, cv2.VideoWriter) == False:\n
		>>> 	out[out_path] = cv2.VideoWriter(\n
		>>> 		os.path.join(out_path, f'{self.name}.avi'), \n
		>>> 		cv2.VideoWriter_fourcc(*'DIVX'), \n
		>>> 		30, \n
		>>> 		(self.ROI_size, self.ROI_size)\n
		>>> 	)\n

		>>> out.write(frame)\n
		>>> self.writers[out_path] = out

		Keyword arguments
		-----------------
		frame : ndarray
			Image that will be written to disk.

		out_path : str
			Path of the output folder where the frame will be written.

		current_frame : int
			Number indicating the position of the frame in the video (starts at zero)

		Return
		------
		frame_path : str
			Junction of the output path (`out_path`) with the name of the frame referencing its position \
				in the video and its file extension. E.g.:
			>>> out_path = 'path/to/output_folder'
			>>> currentframe = 50
			>>> self.image_extension = 'pgm'
			>>> placeholder = 'Frame%05d.%s'
			>>> frame_name = placeholder % (currentframe, self.image_extension)
			>>> print(os.path.join(out_path, frame_name))
			'path/to/output_folder/Frame00050.pgm'
		"""
		# If `frame` is different from `None`
		# If the `out_path` is a folder
		# if the `current_frame` is an integer larger than zero
		if frame is not None and os.path.isdir(out_path) and currentframe >= 0:
			if self.image_extension == 'pgm':
				# Check if you are properly saving as '.pgm': https://aitatanit.blogspot.com/2012/03/writing-pgm-files-in-python-c-and-c.html
				# Convert Frame to Gray
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# Save the frame with the extension that the user wants
			cv2.imwrite(os.path.join(out_path, 'Frame%05d.%s' % (currentframe, self.image_extension)), frame)
			
			# Return the name of the file that was written
			return os.path.join(out_path, 'Frame%05d.%s' % (currentframe, self.image_extension))


class UBFC(Dataset):
	"""This is a `Dataset` subclass. It implements the required methods to use the UBFC dataset."""
	def __init__(self, image_extension: str='png'):
		"""
		Keyword arguments
		-----------------
		image_extension : str
			Filename extension that the frame will have on disk (without the dot before the extension)
		"""
		self.paths = []
		assert isinstance(image_extension, str), f"The `image_extension` parameter must be a string: {image_extension}"
		assert image_extension.isalpha(), f"The `image_extension` parameter must only contain letters: {image_extension}"
		self.image_extension = image_extension

	def map_dataset(self, base_dataset_path: str, video_name_pattern='vid.avi'):
		"""Function that traverses a folder structure to map raw dataset videos. Exclusive to the UBFC indoor dataset.

		Keyword arguments
		-----------------
		base_dataset_path : str
			It is the base dataset folder path (dataset base is the one that must be processed). \
				It must exist and contain the following structure:
				- {base_dataset_path}/{sujeito_folder}/{video_name_pattern}
				- {base_dataset_path}/{sujeito_folder}/ground_truth.txt
		
		video_name_pattern : str
			The variable `video_name_pattern` must be a string with the following formatter or none: (%d). E.g.:
				- 'Frame%05d.pgm' which will be done: pattern % 0 (%05d says the sequence will be five digits \
					long, with the first ones filled with zeros: Frame00000.pgm, Frame00001.pgm, Frame00002.pgm, ...).

		Return
		------
			paths: Iterable
				List of dictionaries containing the following structure: \n
			>>> [
			>>> 	{
			>>> 		'subject_folder_name': 'Subject name or his folder', 
			>>> 		'subdir_folder_name': "Name of the internal folder containing the subject's video", (Empty string in this case)
			>>> 		'video_path': 'Relative or absolute path of the video with filename', 
			>>> 		'ppg_path': 'Relative or absolute path of the PPG with filename'
			>>> 	},
			>>> 	...
			>>> ]
		"""
		# START OF VERIFICATION OF FOLDERS AND BASE FILES

		# The `base_dataset_path` folder should exist as a directory.
		assert os.path.isdir(base_dataset_path), f"The `base_dataset_path` parameter is not a directory: {base_dataset_path}"

		# for each folder inside the base dataset
		for subject_folder_name in os.listdir(base_dataset_path):
			subject_folder = os.path.join(base_dataset_path, subject_folder_name)

			# to assume that the `subject_folder` contains a subject we have to check:
			# if it is a directory
			# If the file 'ground_truth.txt' exists
			# If the file 'vid.avi' exists
			if os.path.isdir(subject_folder) and os.path.isfile(os.path.join(subject_folder, 'ground_truth.txt')):
				video_path = os.path.join(subject_folder, video_name_pattern)
				
				# if the name pattern is valid to the respective video:
				if check_video_pattern(video_path, video_name_pattern, None):
					self.paths.append({
						'subject_folder_name': subject_folder_name, 
						'subdir_folder_name': '', 
						'video_path': video_path,
						'ppg_path': os.path.join(subject_folder, 'ground_truth.txt')
					})

		# naturally order the path list
		self.paths = natsort.natsorted(self.paths, key=itemgetter(*['subject_folder_name']))

	def ppg_flatten(self, pulse_path, out_path) -> None:
		"""Function that opens the MATLAB file, retrives the value of the key named 'pulseOxRecord', flatten the value, \
			overwrites and saves the file as MATLAB in the output path.

		Keyword arguments
		-----------------
		pulse_path : str
			MATLAB file path that contains the PPG.

		out_path : str
			Path of output folder that the handled PPG should be saved as 'pulseOx.mat'.
		"""
		assert os.path.exists(pulse_path), f"The `pulse_path` parameter was not found as a file: {pulse_path}"

		x = np.loadtxt(pulse_path)
		
		savemat(os.path.join(out_path, 'pulseOx.mat'), {
			'bpm': x[1, :],
			'numPulseSample': len(x[1, :]),
			'pulseOxRecord': x[0, :],
			'time_array': x[2, :]
		})

	def save_video(self, frame: np.ndarray, out_path: str, currentframe: int=None):
		"""Function that saves a frame to disk.

		Notes
		-----
		- On disk, the frame can be an image file that has its position in the name, e.g.: 

		>>> cv2.imwrite(os.path.join(out_path, 'Frame%05d.pgm' % currentframe), frame)

		- Also, the frame can be inside a video file, e.g.:\n
			
		>>> out = self.writers.get(out_path, None)
		>>> if out is None or isinstance(out, cv2.VideoWriter) == False:\n
		>>> 	out[out_path] = cv2.VideoWriter(\n
		>>> 		os.path.join(out_path, f'{self.name}.avi'), \n
		>>> 		cv2.VideoWriter_fourcc(*'DIVX'), \n
		>>> 		30, \n
		>>> 		(self.ROI_size, self.ROI_size)\n
		>>> 	)\n

		>>> out.write(frame)\n
		>>> self.writers[out_path] = out

		Keyword arguments
		-----------------
		frame : ndarray
			Image that will be written to disk.

		out_path : str
			Path of the output folder where the frame will be written.

		current_frame : int
			Number indicating the position of the frame in the video (starts at zero)

		Return
		------
		frame_path : str
			Junction of the output path (`out_path`) with the name of the frame referencing its position \
				in the video and its file extension. E.g.:
			>>> out_path = 'path/to/output_folder'
			>>> currentframe = 50
			>>> self.image_extension = 'png'
			>>> placeholder = 'Frame%05d.%s'
			>>> frame_name = placeholder % (currentframe, self.image_extension)
			>>> print(os.path.join(out_path, frame_name))
			'path/to/output_folder/Frame00050.png'
		"""
		# If the `frame` parameter is different from `None`
		# If `out_path` is a folder
		# if `current_frame` is an integer larger than zero
		if frame is not None and os.path.isdir(out_path) and currentframe >= 0:
			if self.image_extension == 'pgm':
				# Check if you are properly saving as '.pgm': https://aitatanit.blogspot.com/2012/03/writing-pgm-files-in-python-c-and-c.html
				# Convert Frame to Gray
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# Save the frame with the extension that the user wants
			cv2.imwrite(os.path.join(out_path, 'Frame%05d.%s' % (currentframe, self.image_extension)), frame)
			
			# Return the name of the file that was written
			return os.path.join(out_path, 'Frame%05d.%s' % (currentframe, self.image_extension))


class MAHNOB(Dataset):
	"""This is a `Dataset` subclass. It implements the required methods to use the MAHNOB dataset."""
	def __init__(self, image_extension: str='png', only_emotions=False):
		"""
		Keyword arguments
		-----------------
		image_extension : str
			Filename extension that the frame will have on disk (without the dot before the extension)
		"""
		self.paths = []
		assert isinstance(image_extension, str), f"The `image_extension` parameter must be a string: {image_extension}"
		assert image_extension.isalpha(), f"The `image_extension` parameter must only contain letters: {image_extension}"
		self.image_extension = image_extension
		self.only_emotions = only_emotions

	def map_dataset(self, base_dataset_path: str, camera_type='color'):
		"""Function that traverses a folder structure to map raw dataset videos. Exclusive to the UBFC indoor dataset.

		Keyword arguments
		-----------------
		base_dataset_path : str
			It is the base dataset folder path (dataset base is the one that must be processed). \
				It must exist and contain the following structure:
				- {base_dataset_path}/{sujeito_folder}/{video filename that will be extracted from the XML}
				- {base_dataset_path}/{sujeito_folder}/.bdf
				- {base_dataset_path}/{sujeito_folder}/session.xml

		camera_type : str
			Type of the cameras that need to be mapped by this method. It could be 'color', 'bw' or 'all'
		
		Return
		------
			paths: Iterable
				List of dictionaries containing the following structure: \n
			>>> [
			>>> 	{
			>>> 		'subject_folder_name': 'Subject name or his folder', 
			>>> 		'subdir_folder_name': "Name of the internal folder containing the subject's video", (Empty string in this case)
			>>> 		'video_path': 'Relative or absolute path of the video with filename', 
			>>> 		'ppg_path': 'Relative or absolute path of the PPG with filename'
			>>> 	},
			>>> 	...
			>>> ]
		"""
		# START OF VERIFICATION OF FOLDERS AND BASE FILES

		required_attr = [
			'feltEmo',
			'feltArsl',
			'feltVlnc',
			'feltCtrl',
			'feltPred'
		]

		# The `base_dataset_path` folder should exist as a directory.
		assert os.path.isdir(base_dataset_path), f"The `base_dataset_path` parameter is not a directory: {base_dataset_path}"

		# for each folder inside the base dataset
		for session_folder_name in os.listdir(base_dataset_path):
			has_required_attr = True
			session_folder = os.path.join(base_dataset_path, session_folder_name)

			session_file = os.path.join(session_folder, 'session.xml')

			# to assume that the `session_folder` contains a subject we have to check:
			# if it is a directory
			# If the file 'session.xml' exists
			# If a '.bdf' file exists
			# If a '.avi' file exists
			if os.path.isfile(session_file) and os.path.isdir(session_folder):
				# https://docs.python.org/2/library/xml.etree.elementtree.html
				session = ET.parse(session_file)
				session_dict = session.getroot().attrib
				session_keys = list(session_dict.keys())

				if self.only_emotions:
					for attr in required_attr:
						has_required_attr &= attr in session_keys

				# if the video was recorded
				if float(session_dict['cutLenSec']) > 0 and has_required_attr:
					subject = session.find('subject')
					if subject is None:
						print(f'Subject returned None: {session_file}')
						continue

					subject_dict = subject.attrib

					if camera_type == 'color':
						video = session.find("./track/[@color='1']")
						if video is None:
							print(f'Video track returned None: {session_file}')
							continue

						video_dict = video.attrib
						assert 'C1' in video_dict['filename'], f'The color video track info was not found on the XML: {video_dict["type"]} from {session_file}'
					elif camera_type == 'bw':
						video = session.find("./track/[@camera='4']")
						if video is None:
							print(f'Video track returned None: {session_file}')
							continue

						video_dict = video.attrib
						assert 'BW' in video_dict['filename'], f'The black and white video track info was not found on the XML: {video_dict["type"]} from {session_file}'
					# TODO tratar o camera_type == 'all'

					gt = session.find("./track/[@type='Physiological']")
					if gt is None:
						print(f'GT track returned None: {session_file}')
						continue

					gt_dict = gt.attrib

					video_path = os.path.join(session_folder, video_dict['filename'])
					gt_path = os.path.join(session_folder, gt_dict['filename'])

					# if the video exists:
					if os.path.isfile(video_path) and os.path.isfile(gt_path) and \
						'.avi' in video_dict['filename'] and '.bdf' in gt_dict['filename']:
						self.paths.append({
							'subject_folder_name': f'subject_{subject_dict["id"]}', 
							'subdir_folder_name': f'session_{session_dict["sessionId"]}', 
							'video_path': video_path,
							'ppg_path': gt_path,
							'session_id': session_dict['cutNr']
						})
					else:
						print('The video or the gt does not exists')
		
		# naturally order the path list
		self.paths = natsort.natsorted(self.paths, key=itemgetter(*['session_id', 'subject_folder_name']))

	def ppg_flatten(self, pulse_path, out_path) -> None:
		"""Function that opens the BDF file, retrives the value of the key named 'EXG3', generates tht PPG wave, \
			retrive the emotions from the XML file and creates the MATLAB file in the output path.

		Keyword arguments
		-----------------
		pulse_path : str
			BDF file path that contains the ECG.

		out_path : str
			Path of output folder that the handled PPG should be saved as 'pulseOx.mat'.
		"""
		assert os.path.exists(pulse_path), f"The `pulse_path` parameter was not found as a file: {pulse_path}"

		bdf = pyedflib.highlevel.read_edf(pulse_path, ch_names='EXG3') # ECG3 (left side of abdomen)
		assert bdf[1][0]['label'] == 'EXG3', f'The channel 34 from the BDF isn\'t the third ECG: {pulse_path}'
		ecg = heartpy.enhance_ecg_peaks(bdf[0][0], bdf[1][0]['sample_rate'])
		# generate PPG
		ppg = ppg_simulate(
			pulse_signal=ecg,
			current_rate=bdf[1][0]['sample_rate'],
			desired_rate=30
		)

		bpm = ppg_to_bpm(ecg, ppg_fps=bdf[1][0]['sample_rate'], stride=4, segment_width=4)

		emotions_path = os.path.join(os.path.dirname(pulse_path), 'session.xml')

		emotions = {}
		if self.only_emotions:
			emotions = {
				'feltEmo': None,
				'feltArsl': None,
				'feltVlnc': None,
				'feltCtrl': None,
				'feltPred': None
			}

			session = ET.parse(emotions_path).getroot().attrib

			for emotion in emotions.keys():
				session_emotion = session.get(emotion, np.nan)
				if session_emotion is np.nan:
					print(f'Emotion {emotion} at the following path was not found: {emotions_path}')
				emotions[emotion] = session_emotion
		
		savemat(os.path.join(out_path, 'pulseOx.mat'), {
			'bpm': bpm,
			'numPulseSample': len(ppg),
			'pulseOxRecord': ppg,
			'ecg': ecg,
			**emotions
		})

	def save_video(self, frame: np.ndarray, out_path: str, currentframe: int=None):
		"""Function that saves a frame to disk.

		Notes
		-----
		- On disk, the frame can be an image file that has its position in the name, e.g.: 

		>>> cv2.imwrite(os.path.join(out_path, 'Frame%05d.pgm' % currentframe), frame)

		- Also, the frame can be inside a video file, e.g.:\n
			
		>>> out = self.writers.get(out_path, None)
		>>> if out is None or isinstance(out, cv2.VideoWriter) == False:\n
		>>> 	out[out_path] = cv2.VideoWriter(\n
		>>> 		os.path.join(out_path, f'{self.name}.avi'), \n
		>>> 		cv2.VideoWriter_fourcc(*'DIVX'), \n
		>>> 		30, \n
		>>> 		(self.ROI_size, self.ROI_size)\n
		>>> 	)\n

		>>> out.write(frame)\n
		>>> self.writers[out_path] = out

		Keyword arguments
		-----------------
		frame : ndarray
			Image that will be written to disk.

		out_path : str
			Path of the output folder where the frame will be written.

		current_frame : int
			Number indicating the position of the frame in the video (starts at zero)

		Return
		------
		frame_path : str
			Junction of the output path (`out_path`) with the name of the frame referencing its position \
				in the video and its file extension. E.g.:
			>>> out_path = 'path/to/output_folder'
			>>> currentframe = 50
			>>> self.image_extension = 'png'
			>>> placeholder = 'Frame%05d.%s'
			>>> frame_name = placeholder % (currentframe, self.image_extension)
			>>> print(os.path.join(out_path, frame_name))
			'path/to/output_folder/Frame00050.png'
		"""
		# If the `frame` parameter is different from `None`
		# If `out_path` is a folder
		# if `current_frame` is an integer larger than zero
		if frame is not None and os.path.isdir(out_path) and currentframe >= 0:
			if self.image_extension == 'pgm':
				# Check if you are properly saving as '.pgm': https://aitatanit.blogspot.com/2012/03/writing-pgm-files-in-python-c-and-c.html
				# Convert Frame to Gray
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# Save the frame with the extension that the user wants
			cv2.imwrite(os.path.join(out_path, 'Frame%05d.%s' % (currentframe, self.image_extension)), frame)
			
			# Return the name of the file that was written
			return os.path.join(out_path, 'Frame%05d.%s' % (currentframe, self.image_extension))


class DemoBasicDataset(MRNirp):
	"""This is a `MR-Nirp` subclass. It implements the `map_dataset` method to use it in the Demo."""
	def __init__(self, image_extension: str='pgm'):
		super().__init__(image_extension)

	def map_dataset(self, base_dataset_path: str, subdir_name=['NIR'], video_name_pattern='Frame%05d.pgm'):
		"""Function that traverses a folder structure to map raw dataset videos.

		Keyword arguments
		-----------------
		base_dataset_path : str
			It is the base dataset folder path (dataset base is the one that must be processed). \
				It must exist and contain the following structure:
				- {base_dataset_path}/{sujeito_folder}/{subdir_name}/{video_name_pattern}
				- {base_dataset_path}/{sujeito_folder}/{subdir_name}/pulseOx.mat
				OR
				- {base_dataset_path}/{sujeito_folder}/{video_name_pattern}
				- {base_dataset_path}/{sujeito_folder}/pulseOx.mat
		
		subdir_name : list or None
			List of folders that are inside (and must be processed) each subject's folder. E.g.:
			- {base_dataset_path}/{sujeito_folder}/RGB/{video_name_pattern} (must have a video inside)
			- {base_dataset_path}/{sujeito_folder}/NIR/{video_name_pattern}
		
		video_name_pattern : str
			The variable `video_name_pattern` must be a string with the following formatter or none: (%d). E.g.:
				- 'Frame%05d.pgm' which will be done: pattern % 0 (%05d says the sequence will be five digits \
					long, with the first ones filled with zeros: Frame00000.pgm, Frame00001.pgm, Frame00002.pgm, ...) or
					vid.avi.

		Return
		------
		paths: Iterable
			List of dictionaries containing the following structure: \n
		>>> [
		>>> 	{
		>>> 		'subject_folder_name': 'Subject name or his folder', 
		>>> 		'subdir_folder_name': "Name of the internal folder containing the subject's video", 
		>>> 		'video_path': 'Relative or absolute path of the video with filename', 
		>>> 		'ppg_path': 'Relative or absolute path of the PPG with filename'
		>>> 	},
		>>> 	...
		>>> ]
		"""
		# START OF VERIFICATION OF FOLDERS AND BASE FILES

		# The `base_dataset_path` folder should exist as a directory.
		assert os.path.isdir(base_dataset_path), f"The `base_dataset_path` parameter is not a directory: {base_dataset_path}"

		if subdir_name is None:
			subdir_name = []

		# for each folder inside the base dataset
		for subject_folder_name in os.listdir(base_dataset_path):
			subject_folder = os.path.join(base_dataset_path, subject_folder_name)

			if os.path.isdir(subject_folder) and os.path.isfile(os.path.join(subject_folder, 'pulseOx.mat')):
				# for each folder within the subject's directory
				for subdir_folder_name in os.listdir(subject_folder):
					item = os.path.join(subject_folder, subdir_folder_name)

					# it verifies if it is a directory and if it is setted to be used
					if os.path.isdir(item) and subdir_folder_name in subdir_name:
						video_path = os.path.join(item, video_name_pattern)
						
						# if the name pattern is valid to the respective video:
						if check_video_pattern(video_path, video_name_pattern, None):
							self.paths.append({
								'subject_folder_name': subject_folder_name, 
								'subdir_folder_name': subdir_folder_name, 
								'video_path': video_path,
								'ppg_path': os.path.join(subject_folder, 'pulseOx.mat')
							})
					elif os.path.isfile(item):
						# if the name pattern is valid to the respective video:
						if check_video_pattern(item, video_name_pattern, None):
							self.paths.append({
								'subject_folder_name': subject_folder_name, 
								'subdir_folder_name': '', 
								'video_path': item,
								'ppg_path': os.path.join(subject_folder, 'pulseOx.mat')
							})

		# naturally order the path list
		self.paths = natsort.natsorted(self.paths, key=itemgetter(*['subject_folder_name']))
		return self.paths


if __name__ == '__main__':
	dataset = MRNirp(image_extension='png')

	# dataset.map_dataset(
	#     base_dataset_path='/home/victorrocha/scratch/desafio2_2021/Datasets/mr_nirp_indoor', 
	#     subdir_name=['RGB'], 
	#     video_name_pattern='Frame%05d.pgm'
	# )

	# demosaic_RGGB_dataset(
	#     dataset=dataset, 
	#     out_dataset_path='/home/victorrocha/scratch/desafio2_2021/Datasets/mr_nirp_indoor',
	#     video_name_pattern='Frame%05d.pgm'
	# )

	dataset.demosaic_video(
		video_path='C:/Users/amysb/OneDrive/ViaLab21/Subject1_still_940/Subject1_still_940/NIR/bottom_face/Frame%05d.pgm',
		out_path='C:/Users/amysb/OneDrive/ViaLab21/teste2', 
		cv2_cvt_code=cv2.COLOR_BAYER_RG2BGR,
		video_name_pattern='Frame%05d.pgm'
	)