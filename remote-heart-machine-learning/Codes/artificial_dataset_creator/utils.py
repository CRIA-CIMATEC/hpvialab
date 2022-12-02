from io import TextIOWrapper
from math import floor
import natsort
import json
import cv2
import os
import re
from tkinter import Canvas
from PIL import Image, ImageTk
from imutils.video import FPS, FileVideoStream
import time
import vlc
from threading import Thread
from argparse import Namespace

def print_log(msg, log_file: TextIOWrapper = None) -> None:
	"""Function that prints and saves a message to the log file.

	Keyword arguments
	-----------------
	msg : str
		String that will be printed and written to the log file.

	log_file : TextIOWrapper
		Open log file (I/O).
	"""
	if log_file:
		log_file.write(f'\n{msg}')
	print(msg)

def make_dir(paths: list or str, log_file: TextIOWrapper = None) -> None:
	"""Function that creates folders if they don't exist.

	Notes
	-----
	- The `os.makedirs` function is used to create the directories.

	Keyword arguments
	-----------------
	paths : list or str
		Path of the folder that to be created.

	log_file : TextIOWrapper
		Open log file (I/O).
	"""
	paths = [paths] if not isinstance(paths, list) else paths
	try:
		for path in paths:
			if not os.path.exists(path):
				os.makedirs(path)
				print_log(f'{path} was created sucessfully', log_file)
	except OSError as e:
		print_log(f'An error occurred while trying to create the directory with the following path: {path}\nError: {e}', log_file)

def check_video_pattern(video_path, video_name_pattern, log_file: TextIOWrapper = None):
	"""Function that will check if the filename pattern works.

	Keyword arguments
	-----------------
	video_path : str
		Filename path that needs to be verified. It contains the `video_name_pattern` at the string.
	video_name_pattern : str
		The variable `video_name_pattern` must be a string with the following formatter or none: (%d). E.g.:
			- 'Frame%05d.pgm' which will be done: pattern % 0 (%05d says the sequence will be five digits \
				long, with the first ones filled with zeros: Frame00000.pgm, Frame00001.pgm, Frame00002.pgm, ...).
	log_file : TextIOWrapper
		Open log file (I/O).

	Return
	------
	video_pattern_exist : bool
		Boolean responsible for indicating whether the pattern applies to the filename or not (exists or not).
	"""
	try:
		# if there is '%d' anywhere in the `video_name_pattern` string
		if len(re.findall("%.*d", video_name_pattern)) > 0:
			# checks if the file formatted with 0 in place of the formatter does not exist
			if os.path.isfile(video_path % 0) == False:
				print_log(f'The `video_name_pattern` parameter formatted with 0 (`video_name_pattern % 0`) isn\'t a file, check if it is correct and if the following file exists: {video_name_pattern % 0}', log_file)
				return False
		else: # if there isn't a string formatting at the `video_name_pattern`
			# it checks if the `video_path` exists
			if os.path.isfile(video_path) == False:
				print_log(f'The `video_path` parameter isn\'t a file, check if it is correct and if the following file exists: {video_path}', log_file)
				return False
			if video_name_pattern not in video_path:
				return False

		# if the file exist
		return True

	except TypeError as type_error:
		print_log(f'TypeError on check_video_pattern({video_path}, {video_name_pattern}): {type_error}', log_file)
	except Exception as e:
		print_log(f'Exception raised on check_video_pattern({video_path}, {video_name_pattern}): {e}', log_file)

def create_features_video(images_path: str, out_path: str, out_name: str=None, 
						out_size=(640, 640), out_fps=30, out_fourcc=cv2.VideoWriter_fourcc(*'DIVX')) -> None:
	"""Function that open writes a video. The input video could be a sequence of images in a folder.

	Keyword arguments
	-----------------
	images_path : str
		Filename or its pattern to the video. It will be readed with `cv2.VideoCapture`. \
			E.g.: '/path/to/sequence%05d.pgm' or '/path/to/video.avi'.
	
	out_path : str
		Output folder path. Doesn't contains the filename.
	
	out_name : str
		Final name of the video file.
	
	out_size : tuple
		Final size of the image within the video (height, width). It must be the same size of the input video.
	
	out_fps : int
		Frame per second used in output video.
	
	out_fourcc : str or cv2.VideoWriter_fourcc
		Video encoding format.
	"""
	cam = cv2.VideoCapture(images_path)
	out_name = 'teste.avi' if out_name == None else out_name
	out = cv2.VideoWriter(os.path.join(out_path, out_name), out_fourcc, out_fps, out_size)

	while True:
		ret, frame = cam.read()

		# if `cv2.VideoCapture` is no longer capturing anything: exit the infinite loop
		if ret == False:
			print_log('The video is over')
			break

		out.write(frame)

	out.release()
	cam.release()

def map_dataset_json(dataset_path, video_name_pattern):
	"""Function that traverses the `dataset_path` folder, searchs for image sequences and the PPG path. 
	It writes the sorted paths in a dictionary and creates a 'dataset_info.json' file inside the `dataset_path` folder.

	Keyword arguments
	-----------------
	dataset_path : str
		Path of the folder that will be mapped.

	video_name_pattern : str
		The variable `video_name_pattern` must be a string with the following formatter or none: (%d). E.g.:
			- 'Frame%05d.pgm' which will be done: pattern % 0 (%05d says the sequence will be five digits \
				long, with the first ones filled with zeros: Frame00000.pgm, Frame00001.pgm, Frame00002.pgm, ...).

	Return
	------
	dataset_json : dict
		Dictionary that will contain the paths to the ROI images and the PPG path. E.g.:
		>>> {
		>>> 	'Subject1': {
		>>> 		'bottom_face': [
		>>> 			'/path/to/bottom_face/frame00000.png',
		>>> 			'/path/to/bottom_face/frame00001.png',
		>>> 			...
		>>> 		],
		>>>			'PPG': '/path/to/PPG/pulseOx.mat',
		>>> 		...
		>>> 	},
		>>> 	'Subject2': ...
		>>>	}
	"""
	dataset_json = {}
	start = video_name_pattern.split('%')[0]
	end = video_name_pattern.split('d')[-1]

	# create a filter base on the `video_name_pattern`
	def frame_filter(filename: str):
		middle = filename.replace(start, '').replace(end, '')
		return filename.startswith(start) and filename.endswith(end) and middle.isdigit()
	
	# for each subject forder
	for subject_folder in natsort.natsorted(os.listdir(dataset_path)):
		# continue another loop in case of a file
		if os.path.isfile(os.path.join(dataset_path, subject_folder)):
			continue

		dataset_json[subject_folder] = {}
		# for each folder or file inside the subject folder
		for item_name in os.listdir(os.path.join(dataset_path, subject_folder)):
			item_path = os.path.join(dataset_path, subject_folder, item_name)

			if os.path.isdir(item_path):
				img_paths = []
				# filter the folder by `video_name_pattern` and sort it
				for img_name in natsort.natsorted(filter(frame_filter, os.listdir(item_path))):
					img_paths.append(os.path.join(item_path, img_name))

				dataset_json[subject_folder][item_name] = img_paths
			elif os.path.isfile(item_path) and item_name == 'pulseOx.mat':
				# add the PPG path to the dict
				dataset_json[subject_folder]['PPG'] = item_path
			else:
				# print other files
				print(f'Unmapped file: {item_path}')

	# it writes the JSON of the dataset to disk
	with open(os.path.join(dataset_path, "dataset_info.json"), 'w') as dataset_json_file:
		json.dump(dataset_json, fp=dataset_json_file, indent=1)

	return dataset_json

class VideoException(Exception):
	"""Class responsible for being launched stopping video processing."""
	def __init__(self, *args: object) -> None:
		super().__init__(*args)

class FrameSequence:
	MediaPlayerEndReached = vlc.EventType.MediaPlayerEndReached
	MediaPlayerTimeChanged = vlc.EventType.MediaPlayerTimeChanged
	def __init__(self, video_path, fps, widget: Canvas, cyclic=True, info_widget=None) -> None:
		# Getting video from webcam
		self.stream = FileVideoStream(video_path, self.transform_frame, queue_size=fps)
		assert self.stream.stream.isOpened(), "Video couldn't be opened"
		# start the FPS timer
		self.fps_counter = FPS()
		self.fps = fps
		self.widget = widget
		self.video_path = video_path
		self.thread_started = False
		self.paused = True
		self.cyclic = cyclic
		self.info_widget = info_widget
		self.end_reached_callback = lambda *args: None
		self.time_changed_callback = lambda *args: None
		self.last_sec = 0
		self.scale = 1
		widget.bind('<Configure>', self.video_set_scale)
		self.thread = None
		self.img_id = None

	def transform_frame(self, frame):
		if frame is not None:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			heigth, width = frame.shape[:2]
			if self.scale != 1:
				frame = cv2.resize(frame, (floor(heigth * self.scale), floor(width * self.scale)))
			return Image.fromarray(frame)
		return None

	def play(self):
		try:
			if self.paused:
				if not self.thread_started:
					self.stream.start()
					self.fps_counter._start = time.time()
					self.thread_started = True
				self.paused = False
				self.thread = Thread(target=self._play, args=())
				self.thread.start()
		except:
			return -1

	def _play(self):
		while True:
			if self.stream.running() and not self.paused:
				start_time = time.time()
				frame = self.stream.read()

				if frame is not None:
					if self.fps_counter._numFrames == 0:
						# Setting the image on the label
						self.img_id = self.widget.create_image(self.widget.winfo_width()//2, self.widget.winfo_height()//2, image=ImageTk.PhotoImage(frame), anchor='center') # Updates the Tkinter window 
					else:
						frame = ImageTk.PhotoImage(frame)
						self.widget.itemconfig(self.img_id, image=frame)
						self.widget.img = frame

					self.widget.update() # Updates the Tkinter window 

					final_time = time.time()
					time_lapsed = round(final_time - start_time, 2)
					sec_wait = 1 / self.fps - time_lapsed
					if sec_wait < 0:
						sec_wait = 0

					time.sleep(sec_wait)
					self.fps_counter.update()
					if self.info_widget is not None:
						self.update_labels()
					if floor((self.get_time() / 1000)) > self.last_sec:
						self.time_changed_callback(Namespace(type=self.MediaPlayerTimeChanged))
						self.last_sec = floor((self.get_time() / 1000))
				else:
					# print('Frame is None')
					pass
			elif self.paused:
				# print('paused')
				break
			elif self.cyclic:
				# print('cicle')
				self.stop()
				self.play()
				break
			else:
				# print('not cicle and stopped')
				self.stream.stop()
				self.end_reached_callback(Namespace(type=self.MediaPlayerEndReached))
				break

	def pause(self):
		if self.thread_started:
			self.paused = True

	def stop(self):
		self.pause()
		# Getting video from webcam
		self.stream = FileVideoStream(self.video_path, self.transform_frame)
		# start the FPS timer
		self.fps_counter = FPS()
		self.thread_started = False
		if self.thread is not None:
			self.thread.join()

	def update_labels(self):
		self.info_widget.children['!label'].config(text=f'Time: {round((self.get_time() / 1000), 2)}', justify='left')
		self.info_widget.children['!label2'].config(text=f'FPS: {round(self.fps_counter._numFrames / (self.get_time() / 1000), 2)}', justify='left')
		self.info_widget.children['!label3'].config(text=f'Frame number: {self.fps_counter._numFrames}', justify='left')

	def video_get_height(self, index=None):
		return self.stream.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
	
	def video_get_width(self, index=None):
		return self.stream.stream.get(cv2.CAP_PROP_FRAME_WIDTH)

	def video_set_scale(self, scale=None):
		video_width, video_height = self.video_get_width(), self.video_get_height()
		if video_width != 0 and video_height != 0:
			width, height = self.widget.winfo_width(), self.widget.winfo_height()
			if width > height:
				scale = height / video_height
			else:
				scale = width / video_width
			self.scale = scale

	def get_time(self):
		if self.fps_counter._start is not None:
			return int((time.time() - self.fps_counter._start) * 1000)
			# return int(self.fps_counter._numFrames * 1000 / self.fps)
		return 0

	def get_length(self):
		return int(self.stream.stream.get(cv2.CAP_PROP_FRAME_COUNT) * 1000 / self.fps)
	
	def event_manager(self):
		return self 

	def event_attach(self, event_type, callback):
		if event_type == self.MediaPlayerEndReached:
			self.end_reached_callback = callback
		elif event_type == self.MediaPlayerTimeChanged:
			self.time_changed_callback = callback

	def event_detach(self, event_type):
		if event_type == self.MediaPlayerEndReached:
			self.end_reached_callback = lambda *args: None
		elif event_type == self.MediaPlayerTimeChanged:
			self.time_changed_callback = lambda *args: None
