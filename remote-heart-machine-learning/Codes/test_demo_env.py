import numpy as np
import cv2

def test_mediapipe_uses():
	"""This function tests the mediapipe's funtions that are called on the project.

	Based on the face_detector.py from the artificial_dataset_creator folder.
	"""
	import sys
	sys.path.append('artificial_dataset_creator')
	from artificial_dataset_creator.face_detector import FaceMeshDetector

	detector = FaceMeshDetector()
	img = cv2.imread('/home/desafio01/Documents/Codes/bio_hmd/Dataset_MR_NIRP/MR-NIRP_Indoor/Subject1_still_940/RGB_demosaiced_antiga/Frame00000.pgm')
	detector.detect_features(img)
	img = detector.lock_frame(img)
	for id, face in detector.faces.items():
		for landmark in face.facial_landmarks:
			for coordinate in landmark.shape:
				img = cv2.circle(img, coordinate, 1, (255, 0, 0), 3)
	cv2.imshow('img', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def test_pandas_use():
	import pandas as pd
	df = pd.read_csv('/home/desafio01/Documents/Codes/bio_hmd/vialab-desafio2-2021/Codes/artificial_dataset_creator/merl_car_downloader/ubfc_structure.csv', sep=',', header=0)
	print(df.head())

def test_natsort_uses():
	import natsort
	import random
	sample_list = list(np.random.randint(0, 20, 15))
	print(sample_list)
	print(natsort.natsorted(sample_list))
	sample_list = list(random.sample(['a', 'b', 'c', 'd', 'e'], 5))
	print(sample_list)
	print(natsort.natsorted(sample_list))

def test_matplotlib__uses():
	import matplotlib.pyplot as plt
	import matplotlib.colors as mcolors
	from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
	from matplotlib.figure import Figure

	plt.plot(np.arange(100))
	plt.show()
	plt.close()

	plt.subplot(211)
	plt.plot(np.arange(100)[::-1])
	plt.subplot(212)
	plt.plot(np.arange(100))
	plt.show()
	plt.close()

	fig = plt.figure()
	fig.suptitle('Sample', fontsize = 20)
	plt.xlabel('Epoch', fontsize = 18)
	plt.ylabel('Loss', fontsize = 16)
	axes = plt.gca()
	
	axes.set_ylim([np.amin(np.arange(100)[20:]), np.amax(np.arange(100)[::-1][20:])])
	plt.plot(np.arange(100), label = "Y")
	plt.plot(np.arange(100)[::-1], label =  "Y val")
	plt.legend(loc="upper right")
	plt.savefig('sample.png')
	print(list(mcolors.BASE_COLORS.keys()) + list(mcolors.TABLEAU_COLORS.keys()))

def check_versions():
	attribute = '__version__'

	packages = {
		'sys': '3.6.10',
		'zipfile38': '0.0.3',
		'numpy': '1.17.0',
		'pandas': '1.1.3',
		'natsort': '7.0.1',
		'heartpy': '1.2.7',
		'scipy': '1.4.1',
		'matplotlib': '3.1.2',
		'cv2': '3.4.15.55',
		'tkinter': '8.6.8',
		'vlc': '3.0.12118',
		'mediapipe': '0.8.0',
		'vidstab': '1.7.3',
		'imutils': '0.5.4',
		'dominate': '2.3.1',
		'future': '0.18.2',
		'skimage': '0.17.2',
		'sklearn': '0.24.0',
		'typing_extensions': '3.10.0.0',
		'visdom': '0.1.8.3',
		'setproctitle': '1.1.10',
		'configobj': '5.0.6',
		'tqdm': '4.23.4',
		'PIL': '7.0.0',
		'torch': '1.4.0',
		'torchvision': '0.5.0',
		'torchsummary': '1.5.1',
		'progress': '1.5',
		'dlib': '19.21.1',
	}

	import importlib

	print('package', 'version', 'current_version')
	for package, version in packages.items():
		try:
			module = importlib.import_module(package)
			current_version = getattr(module, attribute)
			if version not in current_version:
				print(package, version, current_version)
		except:
			print(f'package {package} not available or doesn\'t have the {attribute} attr')

# test_mediapipe_uses()
# test_pandas_use()
# test_natsort_uses()
# test_matplotlib__uses()
check_versions()
