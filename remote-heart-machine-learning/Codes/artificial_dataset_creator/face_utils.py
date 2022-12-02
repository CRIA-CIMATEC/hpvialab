from vidstab.VidStab import VidStab
import numpy as np
import copy
import math
import cv2
import os

class Face:
	"""Class containing facial frames frames and current.
	
	Keyword variables
	-----------------
	self.facial_landmarks : list
		List of `FacialLandmark` objects of the current frame.
	
	self.last_facial_landmarks : list
		List of `FacialLandmark` objects of the previous frame.
	
	self.shape : Iterable
		NUMPY array of tuples that contemplates the coordinates of each detected facial point (X, Y).
	
	self.bottom_face_center : Iterable
		List containing the coordinates of the center of the mouth region (x, y).
	
	self.face_slices_sizes : dict
		Dictionary containing the following structure: {"Name of ROI": (slice_ROI, img_final_size)} 
	"""
	def __init__(self, shape, slices, detector_name):
		"""Receives the coordinates of detected facial points and slices for their respective ROIs.
		"""
		self.facial_landmarks = []
		self.last_facial_landmarks = []
		self.shape = shape
		self.bottom_face_center = None
		self.detector_name = detector_name
		# name of the ROI ; slice it occupies ; final frame size
		self.face_slices_sizes = {
			'bottom_face': (slices[0], 400),
			'right_eye': (slices[1], 100),
			'left_eye': (slices[2], 100),
			# 'middle_face': (slices[3], (500, 276)),
		}

		# Adds each pre-defined ROI in the facial landmarks list
		for name, (face_slice, roi_size) in self.face_slices_sizes.items():
			self.facial_landmarks.append(FacialLandmark(shape[face_slice], name, roi_size, detector_name=self.detector_name))

		# it makes a copy
		# Note: The first prediction will be the comparation of the same landmarks,
		# that is, Euclidean Distance equal to zero
		self.last_facial_landmarks = self.facial_landmarks.copy()

	def update_face(self, shape) -> None:
		"""Method that updates each `FacialLandmark` object with a new shape.
		In addition, updates the latest facial landmarks.
		
		Keyword arguments
		-----------------
		shape : Iterable
			NUMPY array of tuples that contemplates the coordinates of each detected facial point (X, Y).
		"""
		# it makes a copy
		# Note: when this method is called the current landmark and the previous will be different
		self.shape = shape
		self.last_facial_landmarks = []

		# Adds each pre-defined ROI in the facial landmarks list
		for facial_landmark in self.facial_landmarks:
			self.last_facial_landmarks.append(copy.copy(facial_landmark))

			face_slice = self.face_slices_sizes[facial_landmark.name][0]
			facial_landmark.shape = shape[face_slice]

	@staticmethod
	def barrel_distort(src_image, center_x, center_y):
		"""Method that applies the Barrel Distortion in the image considering center X and Y.
		
		Keyword arguments
		-----------------
		src_image : ndarray
			Numpy array of the image in which you want to apply the Barrel Distortion.
		
		center_x : int
			X-axis value that the effect will be applied (referring to src_image).
		
		center_y : int
			Y-axis value that the effect will be applied (referring to src_image).

		Return
		------
		out_img : ndarray
			Image after Barrel Distortion application in the coordinate passed by parameter.
		"""
		# code from Davi Jones: https://stackoverflow.com/questions/64067196/pinch-bulge-distortion-using-python-opencv
		# grab the dimensions of the image
		(h, w, _) = src_image.shape

		# set up the x and y maps as float32
		flex_x = np.zeros((h, w), np.float32)
		flex_y = np.zeros((h, w), np.float32)
		
		scale_y = 1.
		scale_x = 1.
		radius = h / 2 if h < w else w / 2
		amount = -0.4
		elipse_x = 3.8
		elipse_y = 7.1

		# create map with the barrel pincushion distortion formula
		for y in range(h):
				delta_y = scale_y * (y - center_y)
				for x in range(w):
						# determine if pixel is within an ellipse
						delta_x = scale_x * (x - center_x)
						distance = ((delta_x * delta_x) / elipse_x) + ((delta_y * delta_y) / elipse_y)
						if distance >= (radius * radius):
								flex_x[y, x] = x
								flex_y[y, x] = y
						else:
								factor = 1.0
								if distance > 0.0:
										factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), -amount)
								flex_x[y, x] = factor * delta_x / scale_x + center_x
								flex_y[y, x] = factor * delta_y / scale_y + center_y

		# do the remap: this is where the magic happens
		return cv2.remap(src_image, flex_x, flex_y, cv2.INTER_LINEAR)

	def calc_euclidian_dist(self, last_facial_landmark, facial_landmark):
		"""Method that calculates the Euclidean distance from the facial landmarks centers.
		
		Keyword arguments
		-----------------
		last_facial_landmark : FacialLandmark
			`FacialLandmark` class object that represents the landmark detected in the previous frame.
		
		facial_landmark : FacialLandmark
			`FacialLandmark` class object that represents landmark detected in the current frame.

		Return
		------
		euc_dist : float
			Euclidean distance from facial landmarks centers.
		"""
		# Find the center of the lastest landmark: It is only necessary in the first predication
		last_facial_landmark.find_center()
		last_cnt_x, last_cnt_y = last_facial_landmark.get_center()

		facial_landmark.find_center()
		cnt_x, cnt_y = facial_landmark.get_center()
		
		# calculates the euclidian distance from the central point of the current landmark and the previous one.
		euc_dist = math.sqrt(math.pow(cnt_x - last_cnt_x, 2) + math.pow(cnt_y - last_cnt_y, 2))

		return euc_dist

	def calc_threshold(self, facial_landmark):
		"""Method that calculates Threshold according to the relative size of the image.
		That is, if Threshold is exceeded, then the detector will read the cut if the person \
			moves 10% of the relative size of the ROI.
		
		Keyword arguments
		-----------------
		facial_landmark : FacialLandmark
			`FacialLandmark` class object that represents landmark detected in the current frame.

		Return
		------
		threshold: float
			Calculated threshold: 10% of the relative image size.
		"""
		threshold = facial_landmark.get_relative_size() * 0.1

		return threshold

	def crop_bottom_face(self, facial_landmark, face_img, distortion_fixed=None):
		"""Method that cuts on the x-axis of the bottom of the face down. 
		Method required to apply the distort without affecting the other ROIs.
		
		Keyword arguments
		-----------------
		facial_landmark : FacialLandmark
			`FacialLandmark` class object that represents landmark detected in the current frame.
		
		face_img : ndarray
			Numpy array of the image in which you want to cut the bottom of the face down.
		
		distortion_fixed : boolean
			Parameter responsable for determining whether a distortion pinned in the image should be used or not.

		Return
		------
		face_temp : ndarray
			Picture cut on the X-axis with the distort Barrel.
		"""
		# 0.9: leftover margin
		spaced_cnt_y = int((facial_landmark.center_y - facial_landmark.get_relative_size()) * 0.9)
		bottom_face_temp = face_img[spaced_cnt_y:, :].copy()

		# Image created to have the same size of the original image and
		# so that the crop is done correctly
		face_temp = np.zeros_like(face_img)
		
		bottom_face_center = self.bottom_face_center

		if distortion_fixed:
			bottom_face_center = [bottom_face_center[0], spaced_cnt_y + 100]

		face_temp[spaced_cnt_y:, :] = Face.barrel_distort(bottom_face_temp, bottom_face_center[0], bottom_face_center[1] - spaced_cnt_y)
		
		return face_temp

	@staticmethod
	def landmarks_mean(landmark1, landmark2):
		"""Method that makes the average of the objects of the `FacialLandmark` class of the previous frame and the current one.

		Attributes that the average will be applied:
		- `shape` (coordinates of face points);
		- `center_x` (Center of ROI on axis X);
		- `center_y` (Center of ROI on Y axis).

		Intact attributes of the `landmark1` parameter:
		- `name`;
		- `ROI_size`;
		- `out`; (cv2.VideoWriter)
		- `stabilizer`. (VidStab)
		
		Keyword arguments
		-----------------
		facial_landmark : FacialLandmark
			`FacialLandmark` class object that represents Landmark detected in the current frame.
		
		last_facial_landmark : FacialLandmark
			`FacialLandmark` class object that represents the landmark detected in the previous frame.

		Return
		------
		facial_landmark : FacialLandmark
			Facial landmarks resulting from the average.
		"""
		assert landmark1.name == landmark2.name and landmark1.ROI_size == landmark2.ROI_size, "`Name` and `ROI_size` from the landmark parameters must be equal"
		return FacialLandmark(
			shape=np.mean((landmark1.shape, landmark2.shape), axis=0, dtype=int),
			name=landmark1.name,
			ROI_size=landmark1.ROI_size,
			out=landmark1.out,
			center_x=int((landmark1.center_x + landmark2.center_x) / 2),
			center_y=int((landmark1.center_y + landmark2.center_y) / 2),
			stabilizer=landmark1.stabilizer,
			detector_name=landmark1.detector_name
		)

class FacialLandmark:
	"""Class containing data on a ROI.
	
	Keyword variables
	-----------------
	self.shape : Iterable
		NUMPY array of tuples that contemplates the coordinates of the current facial landmark (X, Y).
	
	self.name : str
		Name of ROI.
	
	self.ROI_size : int
		Final size of ROI image.
	
	self.out : cv2.VideoWriter
		Writer instance used to write ROI frames in a file.
	
	self.center_x : int
		Center of ROI on the X axis.
	
	self.center_y : int
		Center of the ROI in Y axis.
	
	self.stabilizer : VidStab
		`Vidstab` instance responsible for applying a frame stabilization by frame or entire video.
	"""
	def __init__(self, shape, name, ROI_size, out=None, center_x=None, center_y=None, stabilizer=None, detector_name=None):
		self.shape = shape
		self.name = name
		self.ROI_size = ROI_size
		self.center_x = center_x
		self.center_y = center_y
		self.out = out
		self.stabilizer = stabilizer
		self.detector_name = detector_name

	def find_center(self) -> None:
		"""Method that receives the coordinates of a part of the face and calculates the average \
			point of each of the axes of these coordinates."""
		self.center_x = np.mean(self.shape[:, 0])
		self.center_y = np.mean(self.shape[:, 1])

	def get_relative_size(self, shape=None):
		"""Method that calculates the relative size of ROI in relation to the X-axis.
		The relative size varies when the person moves away or approaches the camera.
		
		Keyword arguments
		-----------------
		shape : Iterable
			NUMPY array of tuples that contemplates the coordinates of the current facial landmark (X, Y).

		Return
		------
		relative_size : float
			Relative ROI size.
		"""
		shape = self.shape if shape is None else shape
		assert isinstance(shape, np.ndarray), 'Method: `FacialLandmark.get_relative_size`\n`shape` must be an instance of `np.array`'
		
		minX = np.amin(shape[:, 0])
		maxX = np.amax(shape[:, 0])
		
		size = 0.5
		if self.detector_name == 'SVM' and 'eye' in self.name:
			size = 0.8

		# Calculates half the size of the region of interest in the X-axis
		relative_size = int(size * (maxX - minX))
		return relative_size

	def get_center(self):
		"""Method that returns a tuple with the value of the center of the ROI (X, Y).
		
		Return
		------
		(self.center_x, self.center_y) : tuple
			Tuple with the value of the center of ROI (X, Y).
		"""
		return (self.center_x, self.center_y)

	def crop_feature(self, img, center_x=None, center_y=None, shape=None):
		"""Method that calculates and cuts ROI according to the relative size.
		
		Keyword arguments
		-----------------
		img : ndarray
			Numpy array of the image that has the ROI that will be cropped.
		
		center_x : int
			Center of ROI on the X axis.
		
		center_y : int
			Center of the ROI in Y axis.
		
		shape : Iterable
			NUMPY array of tuples that contemplates the coordinates of the current facial landmark (X, Y).

		Return
		------
		out_img : ndarray
			Image after cutout relative to the detected points.
		"""
		# Calculate the middle point of the coordinates
		self.find_center()

		center_x = self.center_x if center_x is None else center_x
		center_y = self.center_y if center_y is None else center_y
		shape = self.shape if shape is None else shape
		assert isinstance(center_x, (float, int, np.float, np.int)), 'Method: `FacialLandmark.crop_feature`\n`center_x` must be an instance of `int` or `float`'
		assert isinstance(center_y, (float, int, np.float, np.int)), 'Method: `FacialLandmark.crop_feature`\`ncenter_y` must be an instance of `int` or `float`'
		assert isinstance(shape, np.ndarray), 'Method: `FacialLandmark.crop_feature`\n`shape` must be an instance of `np.array`'

		if self.name == 'middle_face':
			rel_land = [
				min(shape[:, 0]), min(shape[:, 1]),
				max(shape[:, 0]), max(shape[:, 1]),
			]
		else:
			margin = 0
			relative_size = self.get_relative_size(shape)

			# calculates coordinates according to the center of each axis and with the relative size
			# [x, y, x+w, y+h] or [x_inicial, y_inicial, x_final, y_final]
			rel_land = [
					int(center_x) - relative_size - margin, int(center_y) - relative_size - margin, 
					int(center_x) + relative_size + margin, int(center_y) + relative_size + margin
					]

		# cuts the image from the beginning to the end of y and x
		return img[rel_land[1] : rel_land[3], rel_land[0] : rel_land[2]], rel_land

	def update_video(self, out_path, frame) -> None:
		"""Method that updates the ROI video that is being written (creates a buffer if it has not yet happened).
		
		Keyword arguments
		-----------------
		out_path : str
			Video output path.
		
		frame : ndarray
			Numpy array of the image that will be written in the buffer.
		"""
		if self.out is None or isinstance(self.out, cv2.VideoWriter) == False:
			# print('out created')
			self.out = cv2.VideoWriter(
				os.path.join(out_path, f'{self.name}.avi'), 
				cv2.VideoWriter_fourcc(*'DIVX'), 
				30, 
				(self.ROI_size, self.ROI_size)
			)

		self.out.write(frame)

	def stabilize_frame(self, frame, current_frame_number, border_size=-5):
		"""Method that makes a stabilization frame by frame using Vidstab.

		Notes
		-----
		- If the amount of frames passed to this method is less than 30:
			- Return the frame without stabilization with the edges cutted.
		
		Keyword arguments
		-----------------
		frame : ndarray
			Numpy array of the image that will be stabilized.
		
		current_frame_number : int
			Counter of how many frames were passed to the method.
		
		border_size : int
			Negative value for how much edge should be removed from the final image.

		Return
		------
		frame or stab : ndarray
			Numpy Array of the stabilized image.
		"""
		if self.stabilizer is None:
			self.stabilizer = VidStab()

		stab = self.stabilizer.stabilize_frame(input_frame=frame, border_size=border_size)
		
		if current_frame_number < 30:
			return frame[-border_size:border_size, -border_size:border_size]
			
		return stab

	def __copy__(self):
		"""Required method to make copies of the object.

		Return
		------
		facial_landmark : FacialLandmark
			New `Faciallandmark` instance using the attributes of the current object.
		"""
		return type(self)(self.shape, self.name, self.ROI_size, None)
