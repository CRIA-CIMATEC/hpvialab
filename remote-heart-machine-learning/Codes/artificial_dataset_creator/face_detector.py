from mediapipe.python.solutions.face_mesh import FaceMesh
from mesh_points import MESH_ANNOTATIONS
from imutils import face_utils
from face_utils import Face
import numpy as np
import cv2
import abc

class FaceDetector(metaclass=abc.ABCMeta):
	"""Class that detects the facial landmarks.

	Subclass detector types
	-----------------------
		- 'SVM'
		- 'FaceMesh'
	
	Keyword variables
	-----------------
	self.detector_type : str
		Facial landmarks detector name that should be used.
	
	self.faces : dict
		Dictionary that contains detected faces, it has `Face` objects as values and integers as keys
		
	self.detector : Object
		Detector object used.
	
	quant_fired_exceptions : static int
		Number of thrown exceptions during execution.
	
	max_quant_exceptions : static int
		Maximun number of exceptions thrown during execution.
	"""
	quant_fired_exceptions = 0
	max_quant_exceptions = 10

	def __init__(self):
		"""
		Subclass detector types
		-----------------------
			- SVMDetector
			- FaceMeshDetector
		"""

		self.faces = {}

		# reset the counter when an instance of the class is created
		FaceDetector.quant_fired_exceptions = 0

	@abc.abstractmethod
	def detect_features(self, img) -> None:
		"""Method that receives an image and detects the ROIs.
		
		Keyword arguments
		-----------------
		img : Iterable
			Numpy array of the image in which you want to detect the facial landmarks.
		"""
		raise NotImplementedError

	@abc.abstractmethod
	def normalize_landmarks(self, face, img_shape=None):
		"""Method that transforms landmarks into coordinates (x, y) according to the detector.
		
		Keyword arguments
		-----------------
		face : Object
			List of landmarks returned by the detector.
		
		img_shape : tuple
			Three-position tuple containing the shape of a video image (height, width, channels).

		Return
		------
		coordinates : list
			List of coordinates handled according to the detector (x, y).
		"""
		raise NotImplementedError

	@staticmethod
	def update_exceptions():
		"""Method that updates the amount of exceptions thrown during execution. It also verifies \
			if the amount reached its maximum.
		
		Return
		------
		raise_exception : bool
			Boolean indicating if the exception should be thrown or not.
		"""
		FaceDetector.quant_fired_exceptions += 1

		# if the amount reached its maximum during the execution:
		if FaceDetector.quant_fired_exceptions >= FaceDetector.max_quant_exceptions:
			return True # it must be a `VideoException`
		return False # there should be no `VideoException`

	def lock_frame(self, img):
		"""Method that lock the eyes of the detected subject in the image.

		Notes
		-----
		- This method needs a face with the facial landmarks detected
		- Only the eyes of the first detected face will be used to lock the frame
		
		Keyword arguments
		-----------------
		img : ndarray
			Numpy array of the image that you want to lock the frame.

		Return
		------
		img : ndarray
			Numpy array of the locked image based on the landmarks of the eyes.
		"""
		# code based on: https://github.com/vaibhavhariaramani/imutils/blob/016917e0c96178764ef8cda78462a8f23afc9a7b/imutils/face_utils/facealigner.py
		desiredFaceWidth = img.shape[1]
		desiredFaceHeight = img.shape[0]

		# compute the center of mass for each eye
		leftEyeCenter = np.array(self.faces[0].facial_landmarks[2].shape).mean(axis=0).astype("int")
		rightEyeCenter = np.array(self.faces[0].facial_landmarks[1].shape).mean(axis=0).astype("int")

		angle = 0
		scale = 1

		# compute center (x, y)-coordinates (i.e., the median point)
		# between the two eyes in the input image
		eyesCenter = (int((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
					(int(leftEyeCenter[1] + rightEyeCenter[1]) // 2))
		# grab the rotation matrix for rotating and scaling the face
		M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
		# update the translation component of the matrix
		# the eyes will be locked at the middle of the X axis and at the top of the Y axis
		tX = desiredFaceWidth * 0.5
		tY = desiredFaceHeight * 0.2 # desiredLeftEye[1]
		M[0, 2] += (tX - eyesCenter[0])
		M[1, 2] += (tY - eyesCenter[1])

		# it saves the affine parameters to update the coordinates of the facial landmarks.
		affine_params = (int(tX - eyesCenter[0]), int(tY - eyesCenter[1]))

		# apply the affine transformation
		(w, h) = (desiredFaceWidth, desiredFaceHeight)
		img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

		# crop borders from bottom and right space
		img = img[:affine_params[1], :] if affine_params[1] < 0 else img
		img = img[:, :affine_params[0]] if affine_params[0] < 0 else img

		# updates the facial landmarks
		for face_key, face in self.faces.items():
			for landmark_index, landmark in enumerate(face.facial_landmarks):
				for shape_index, shape in enumerate(landmark.shape):
					self.faces[face_key].\
						facial_landmarks[landmark_index].\
							shape[shape_index] = [
								shape[0] + affine_params[0], 
								shape[1] + affine_params[1]
							]

			bottom_face_center = self.faces[face_key].bottom_face_center
			self.faces[face_key].bottom_face_center = [
				bottom_face_center[0] + affine_params[0],
				bottom_face_center[1] + affine_params[1],
			]
		
		return img

	@classmethod
	def __subclasshook__(cls, subclass):
		"""Method that overwrites the `__subclasshook__()` method to customize `issubclass()`.

		Notes
		-----
		- The `subclass` object must have implemented the `detect_features` and `normalize_landmarks` methods.
		
		Keyword arguments
		-----------------
		subclass : object
			Object that needs to be checked.

		Return
		------
		issubclass : bool
			Boolean that indicates if an object has implemented the abstract methods.
		"""
		return (hasattr(subclass, 'detect_features') and 
				callable(subclass.detect_features) and 
				hasattr(subclass, 'normalize_landmarks') and 
				callable(subclass.normalize_landmarks) or 
				NotImplemented)

class SVMDetector(FaceDetector):
	'''SVM class to detect facial landmarks'''
	def __init__(self, path_weigths):
		"""
		Keyword arguments
		-----------------
		path_weigths : str
			Path of dlib predictor weights.
		"""
		super().__init__()
		
		import dlib

		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(path_weigths)

	def detect_features(self, img):
		'''Method to detect features'''
		slices = [
			slice(2, 15),
			slice(36, 42),
			slice(42, 48)
		]

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Gray Filter
		rects = self.detector(img, 0)
		for (i, rect) in enumerate(rects):
			shape = self.predictor(gray, rect)
			shape = self.normalize_landmarks(shape)
			
			if i in self.faces:
				self.faces[i].update_face(shape)
			else:
				self.faces[i] = Face(shape, slices, 'SVM')
			self.faces[i].bottom_face_center = [
				int((self.faces[i].shape[51][0] + self.faces[i].shape[33][0]) / 2),
				int((self.faces[i].shape[51][1] + self.faces[i].shape[33][1]) / 2),
			]

		# delete previous faces
		for j in range(len(self.faces)):
			if j > i:
				del self.faces[j]
	
	def normalize_landmarks(self, face, img_shape=None):
		return face_utils.shape_to_np(face)

class FaceMeshDetector(FaceDetector):
	'''FaceMesh class to detect facial landmarks'''
	def __init__(self):
		super().__init__()
		self.detector = FaceMesh(max_num_faces=1)

	def detect_features(self, img):
		slices = [
			slice(0, 20),
			slice(20, 36),
			slice(36, 52),
			slice(99, 263),
		]

		rects = self.detector.process(img)
		if rects.multi_face_landmarks:
			for (i, rect) in enumerate(rects.multi_face_landmarks):
				shape = self.normalize_landmarks(rect.landmark, img.shape)
				assert shape is not None, "Shape cannot be `None`"
				shape = np.array(shape)
				if i in self.faces:
					self.faces[i].update_face(shape)
				else:
					self.faces[i] = Face(shape, slices, 'FaceMesh')
				self.faces[i].bottom_face_center = [
					int((self.faces[i].shape[57][0] + self.faces[i].shape[96][0]) / 2),
					int((self.faces[i].shape[57][1] + self.faces[i].shape[96][1]) / 2),
				]
		else:
			raise Exception("No face was detected")
	
	def normalize_landmarks(self, face, img_shape=None):
		coordinates = []
		for id in [*MESH_ANNOTATIONS['bottom_face'], 
					*MESH_ANNOTATIONS['rightEye'], 
					*MESH_ANNOTATIONS['leftEye'],
					*MESH_ANNOTATIONS['mouth'],
					*MESH_ANNOTATIONS['nose'],
					# *MESH_ANNOTATIONS['middle_face']
					]:
			assert img_shape is not None, "The `img_shape` parameter shouldn't be `None` when the `FaceDetector` is a `FaceMeshDetector`."
			landmark = face[id]
			h, w, _ = img_shape
			coordinates.append([int(landmark.x * w), int(landmark.y * h)])
		return coordinates

if __name__ == '__main__':
	detector = FaceMeshDetector() # 'shape_predictor_68_face_landmarks.dat'
	img = cv2.imread('/home/desafio01/Documents/Codes/bio_hmd/Dataset_MR_NIRP/MR-NIRP_Indoor/Subject3_still_940/IR_20s/Frame00000.pgm')
	detector.detect_features(img)
	locked = detector.lock_frame(img)
	cv2.imshow('img', locked)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	pass