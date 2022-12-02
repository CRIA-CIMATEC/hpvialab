# -*- coding: utf-8 -*-
from utils import VideoException, check_video_pattern, make_dir, print_log
from face_detector import FaceDetector, FaceMeshDetector, SVMDetector
from dataset_classes import Dataset
import dataset_classes
from datetime import datetime
from progress.bar import Bar
from io import TextIOWrapper
from vidstab import VidStab
from face_utils import Face
import numpy as np
import argparse
import logging
import json
import cv2
import os

import sys
sys.path.append('../deep_motion_mag')
from magnet import deepmag_model

# link from the SVM weight: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
path_weigths = 'shape_predictor_68_face_landmarks.dat'

def process_video(video_path, out_path, dataset: Dataset, detector_type='FaceMesh', save_log=True, save_videos=True, 
				realtime=False, use_threshold=True, distortion_fixed=False, use_frame_stabilizer=False, use_eyes_lock=False, 
				use_video_stabilizer=False, threshold_modes: str or dict='mean', use_magnification=False):
	"""Function that receives the video's path and process it frame by frame with the facial detector. 
	The detected region of interest (ROI) will receive some preprocessing, like stabilization, cutting and
	barrel distortion (in case of the ROI "bottom face").
	
	ROIs
	----
	- Bottom face;
	- Left and right eye;
	- Middle-face.
	
	Threshold modes
	---------------
	- 'mean': When the threshold is exceeded, this mode averages the coordinates of the last and the current facial landmark (smooth transition);
	- 'replace': When the threshold is exceeded, this mode replaces the coordinates of the last with the current facial landmark (sudden transition).

	Keyword arguments
	-----------------
	video_path : str
		Video path that will be passed to the `cv2.VideoCapture` method (video that must be processed). It can be a sequence of \
			images in a folder (name pattern example: Frame%05d.pgm) or it could be a `cv2` compatible video.

	out_path : str
		Path of the output folder where you want to save the ROI videos (a folder will be created for each ROI inside the `out_path`).

	dataset : Dataset
		The `dataset` object must belong to a class that enherit the class `Dataset`. The `dataset` object class must have implemented \
			at least the `save_video` method. If the `use_video_stabilizer` parameter is set to `True`, then the `dataset` must have a \
				string variable called `image_extension`, like 'png' or 'pgm'.
	
	detector_type : str
		Detector type that will be used to detect the facial landmarks. It can be 'SVM' or 'FaceMesh' (more stable and with more facial points).
	
	save_log : boolean
		Parameter responsable for determining whether the log should be saved or not. The execution log will be saved inside the \
			`out_path` folder as 'log.txt'.
	
	save_videos : boolean
		Parameter responsable for determining whether the ROI videos should be saved or not.
	
	realtime : boolean
		Parameter responsable for determining whether the processed video should be displayed in real time with `cv2.imshow` or not. Press Q on keyboard to exit.
	
	use_threshold : boolean
		Parameter responsable for determining whether the threshold method should be used or not.
	
	distortion_fixed : boolean
		Parameter responsable for determining whether a distortion pinned in the image should be used or not.
	
	use_frame_stabilizer : boolean
		Parameter responsable for determining whether the frame by frame stabilization method should be used or not.
	
	use_eyes_lock : boolean
		Parameter responsable for determining whether the eyes should be fixed in the image or not.
	
	use_video_stabilizer : boolean
		Parameter responsable for determining whether the entire video stabilization (`VidStab.stabilize`) method should be used or not.
		The `use_video_stabilizer` parameter saves the video as a single file because it uses the `VideoWriter` method.
		When the `use_video_stabilizer` parameter is set to `True`, then it saves the video's name with the '_vidstab' suffix.
		If the `use_video_stabilizer` parameter is set to `True`, then the `dataset` must have a string variable called `image_extension`, like 'png' or 'pgm'.
	
	threshold_modes : str or dict
		The `threshold_modes` parameter will indicate which threshold to use for each facial landmark.
		If the `threshold_modes` parameter is a string, then a dictionary filled with that string will be created for each ROI.
		If the `threshold_modes` parameter is a dictionary, then his keys must reference each and every ROI, in addition, it values \
			must be a threshold mode.

	Returns
	------- 
	video_json : dict
		Dictionary that contains ROI names as keys and a list of paths as values. E.g.:
		>>> {
		>>> 	'bottom_face': ['path1/to/frame0.png', 'path1/to/frame1.png', ...],
		>>> 	'right_eye': ['path2/to/frame0.png', 'path2/to/frame1.png', ...],
		>>> 	'left_eye': ['path3/to/frame0.png', 'path3/to/frame1.png', ...],
		>>> 	'middle_face': ['path4/to/frame0.png', 'path4/to/frame1.png', ...],
		>>> }

	Raises
	------
	VideoException
		There is a failure limit that the face detector can have. If it is exceeded, a `VideoException` will be thrown, \
			stopping the video processing. There is a failure limit that the entire function can have. If it is exceeded, \
			a `VideoException` will be thrown, stopping the video processing.
			
	Exception
		Occurs when the exception thrown was not mapped by the implementation.
	"""
	
	# Function used inside the `process_video` function
	def close_all_resources(face_detector: FaceDetector, cam: cv2.VideoCapture, log_file: TextIOWrapper, 
							bar: Bar, save_videos=False, save_log=False):
		"""Function that closes all open I/O resources.

		Keyword arguments
		-----------------
		face_detector : FaceDetector
			`FaceDetector` instance that contains the `VideoWriter`s that need to be closed.
		
		cam : cv2.VideoCapture
			`cv2.VideoCapture` instance that need to be closed.
		
		log_file : TextIOWrapper
			Opened log file (I/O).
		
		bar : Bar
			Progress bar that need to be terminated.
		
		save_videos : boolean
			Parameter responsable for determining whether the ROI videos should be saved or not.
		
		save_log : boolean
			Parameter responsable for determining whether the log should be saved or not. The execution log will be saved inside the \
				`out_path` folder as 'log.txt'.
		"""
		if save_videos:
			for i in range(len(face_detector.faces)):
				for facial_landmark in face_detector.faces[i].facial_landmarks:
					if facial_landmark.out is not None:
						if facial_landmark.out.isOpened():
							facial_landmark.out.release()

		if cam:
			if cam.isOpened():
				# When video is done, release the `cv2.VideoCapture` object
				cam.release()

		if save_log:
			log_file.close()
		
		if bar:
			bar.finish()

		# Close all the frames
		cv2.destroyAllWindows()
	
	assert issubclass(dataset.__class__, Dataset), "The `dataset` object must belong to a class that enherit the class `Dataset`."
	
	if use_video_stabilizer:
		assert isinstance(dataset.image_extension, str), f"The `dataset.image_extension` variable must be a string: {dataset.image_extension}"
		assert dataset.image_extension.isalpha(), f"The `dataset.image_extension` variable must only contain letters (like 'png' or 'pgm'): {dataset.image_extension}"

	cam = cv2.VideoCapture(video_path)
	video_json = {}
	log_file = None
	if save_videos or save_log:
		make_dir(out_path, log_file)

	threshold_mode = ''
	landmark_names = ['bottom_face', 'right_eye', 'left_eye']  # , 'middle_face'
	threshold_types = ['mean', 'replace']
	
	# `threshold_modes` treatment and validation
	if threshold_modes:
		assert isinstance(threshold_modes, str) or isinstance(threshold_modes, dict), \
			f"The `threshold_modes` parameter of the `process_video` function must be a string or a dictionary, type(threshold_modes): {type(threshold_modes)}"

		if isinstance(threshold_modes, str):
			threshold_mode = str(threshold_modes).lower()
			assert threshold_mode in threshold_types, f"The `threshold_modes` parameter value is not part of the threshold types ({threshold_types}), parameter value: {threshold_mode}"
			threshold_modes = {}
			for facial_landmark in landmark_names:
				threshold_modes[facial_landmark] = threshold_mode

		if isinstance(threshold_modes, dict):
			assert sorted(list(threshold_modes.keys())) == sorted(landmark_names), \
				f"The `threshold_modes` parameter keys should look like this: {landmark_names}.\nlist(threshold_modes.keys()): {list(threshold_modes.keys())}"
			for value in threshold_modes.values():
				assert value in threshold_types, f"The `threshold_modes` parameter value is not part of the threshold types ({threshold_types}), parameter value: {value}"

	if save_log:
		log_file = open(os.path.join(out_path, "log.txt"), "w+")
		log_file.write(f"""\
			\rVideo_path: {video_path}\
			\rDetector_type: {detector_type}\
			\rUse_Threshold: {use_threshold}\
			\ruse_eyes_lock: {use_eyes_lock}\
			\ruse_frame_stabilizer: {use_frame_stabilizer}\
			\ruse_video_stabilizer: {use_video_stabilizer}\
			\rDistortion_Fixed: {distortion_fixed}\
			\rThreshold Modes: {threshold_modes}\n""")
	
	# setting face detector type
	detector_type = detector_type.lower()

	# creating conditional to use the selected face detector
	if detector_type == "svm":
		face_detector = SVMDetector(path_weigths)
	elif detector_type == "facemesh":
		face_detector = FaceMeshDetector()
	else:
		raise VideoException(f"The `detector_type` parameter isn't valid. It can be 'SVM' or 'FaceMesh'. `detector_type`: {detector_type}")

	currentframe = 0
	status = ''
	euc_dist = 0
	bar = None
	fail_detector_count = 0
	last_frame = {}
	magnify_dict = {}

	def interval_mapping(image, from_min, from_max, to_min, to_max):
		# map values from [from_min, from_max] to [to_min, to_max]
		# image: input array
		from_range = from_max - from_min
		to_range = to_max - to_min
		scaled = np.array((image - from_min) / float(from_range), dtype=float)
		return to_min + (scaled * to_range)

	while(True):
		try:
			ret, frame = cam.read()

			# if the capture is done: break the `while(True)` loop
			if ret == False:
				break

			try:
				# it detects the faces and the landmarks present in the frame
				face_detector.detect_features(frame)

				# if the detector succeeds: it resets the detector's fault counter
				fail_detector_count = 0
			except Exception as e:
				fail_detector_count += 1

				# it updates the number of exceptions thrown at the end of the execution
				FaceDetector.quant_fired_exceptions += 1

				print_log(f"\nDetector failed to detect facial landmarks.\nNumber of detector failures: {fail_detector_count}\n", log_file)
				if fail_detector_count >= FaceDetector.max_quant_exceptions:
					# it removes the added exceptions as they will count as `VideoException`
					FaceDetector.quant_fired_exceptions -= FaceDetector.max_quant_exceptions
					raise VideoException(f'`FaceDetector` sub-class `detect_features` method exceeded the maximum number of times ({FaceDetector.max_quant_exceptions}) it can fail: {e}')

			if use_eyes_lock:
				frame = face_detector.lock_frame(frame)

			face = frame.copy() if realtime else frame
			# for each face present in the frame
			for i in range(len(face_detector.faces)):
				zipped_landmarks = zip(face_detector.faces[i].facial_landmarks, face_detector.faces[i].last_facial_landmarks)
				# for each landmark of the face
				for j, (facial_landmark, last_facial_landmark) in enumerate(zipped_landmarks):
					if use_video_stabilizer and currentframe == 0:
						facial_landmark.ROI_size += 10

					# it gets the final size that the ROI should be after doing the resize
					final_img_size = facial_landmark.ROI_size

					# it calculates the Euclidian distance from the center point of the current and the previous landmark 
					euc_dist = face_detector.faces[i].calc_euclidian_dist(last_facial_landmark, facial_landmark)

					# it calculates the `threshold` according to the relative size of the image
					threshold = face_detector.faces[i].calc_threshold(facial_landmark)
					
					# if the Euclidian distance (`euc_dist`) is less than the `threshold`:
					# 	if threshold_mode is equal to 'mean':
					# 		You will have the average of the previous and current landmark
					# 	elif threshold_mode is equal to 'replace':
					# 		You will have the values of the previous landmark 
					# else:
					# 	Keep current landmark
					if euc_dist <= threshold and use_threshold:
						threshold_mode = threshold_modes[facial_landmark.name]
						if threshold_mode == 'mean':
							facial_landmark = face_detector.faces[i].facial_landmarks[j] = Face.landmarks_mean(facial_landmark, last_facial_landmark)
						elif threshold_mode == 'replace':
							facial_landmark.shape = face_detector.faces[i].facial_landmarks[j].shape = last_facial_landmark.shape
							facial_landmark.center_x = face_detector.faces[i].facial_landmarks[j].center_x = last_facial_landmark.center_x
							facial_landmark.center_y = face_detector.faces[i].facial_landmarks[j].center_y = last_facial_landmark.center_y
						else:
							raise Exception(f'`threshold_mode` {threshold_mode} unrecognized: must be "mean" or "replace"')

					if facial_landmark.name == 'bottom_face':
						# it cuts the bottom of the face down 
						face_temp = face_detector.faces[i].crop_bottom_face(facial_landmark, face, distortion_fixed)
						cut, rel_land = facial_landmark.crop_feature(face_temp)
					else:
						# it cuts the frame at the detected position of the landmark
						cut, rel_land = facial_landmark.crop_feature(face)
					
					cut = np.array(cut)
					size = cut.shape
					
					# it adds edge to remove after stabilization
					border = 10 if use_frame_stabilizer else 0

					if isinstance(final_img_size, int):
						cut = cv2.resize(cut, (final_img_size + border, final_img_size + border), interpolation=cv2.INTER_CUBIC)
					elif isinstance(final_img_size, tuple):
						cut = cv2.resize(cut, (final_img_size[0] + border, final_img_size[1] + border), interpolation=cv2.INTER_CUBIC)
					else:
						raise Exception(f'The `final_img_size` variable must be an integer or a tuple:\nLandmark: {facial_landmark.name}\nfinal_img_size: {final_img_size}')
					
					if use_frame_stabilizer:
						cut = facial_landmark.stabilize_frame(cut, int(-border/2))
					
					if use_magnification:
						magnify = magnify_dict.get(facial_landmark.name, None)
						if magnify is None:
							magnify = deepmag_model(
								config_spec='../deep_motion_mag/configs/configspec.conf', 
								config_file='../deep_motion_mag/configs/o3f_hmhm2_bg_qnoise_mix4_nl_n_t_ds3.conf', 
								phase='run',
								frame_shape=cut.shape
							)
							magnify_dict[facial_landmark.name] = magnify
						last_roi_frame = last_frame.get(facial_landmark.name, None)
						# the `cut` variable will always become the last frame
						last_frame[facial_landmark.name] = cut.copy()
						# if the frame isn't the first
						if last_roi_frame is not None:
							cut = magnify(last_roi_frame, cut, 10) 
							cut = interval_mapping(cut, -1, 1, 0, 255)	
						del last_roi_frame
					
					if realtime:
						'''Method to show facial landmarks in real time'''

						pos = 100 + 100*list(face_detector.faces[i].face_slices_sizes.keys()).index(facial_landmark.name)
						cv2.imshow(f'face{i+1}_{facial_landmark.name}_relative', cut)
						frame = cv2.putText(frame, f"Eucl. dist. {facial_landmark.name}: {euc_dist:.2f}", (0, pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4)
						frame = cv2.putText(frame, f"size: {size} ; thr: {threshold:.2f}", (0, pos+50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4)

						frame = cv2.rectangle(
							frame, 
							(rel_land[0], rel_land[1]), 
							(rel_land[2], rel_land[3]),
							(0, 0, 255),
							2)

						for j, point in enumerate(facial_landmark.shape):
							frame = cv2.circle(frame, (point[0], point[1]), 1, (0, 0, 255), -1)

						# frame = cv2.circle(frame, tuple(face_detector.faces[i].bottom_face_center), 1, (0, 0, 255), -1)
						frame = cv2.circle(frame, (int(facial_landmark.center_x), int(facial_landmark.center_y)), 1, (0, 0, 255), -1)

					# save the cutted frame
					if save_videos:
						status = f'Saving crops sucessfully'
						if isinstance(facial_landmark.ROI_size, int):
							assert cut.shape[:2] == (facial_landmark.ROI_size, facial_landmark.ROI_size), f"`Cut` with wrong size: {cut.shape[:2]} {(facial_landmark.ROI_size, facial_landmark.ROI_size)}"
						elif isinstance(final_img_size, tuple):
							assert cut.shape[:2] == (facial_landmark.ROI_size[1], facial_landmark.ROI_size[0]), f"`Cut` with wrong size: {cut.shape[:2]} {(facial_landmark.ROI_size[1], facial_landmark.ROI_size[0])}"
						make_dir(os.path.join(out_path, facial_landmark.name), log_file)

						frame_path = dataset.save_video(
							frame=cut,
							out_path=os.path.join(out_path, facial_landmark.name),
							currentframe=currentframe,
						)
						
						if frame_path and isinstance(frame_path, str):
							# it adds the frame path to the corresponding dictionary key
							landmark_list = video_json.get(facial_landmark.name, [])
							landmark_list.append(frame_path)
							video_json[facial_landmark.name] = landmark_list

			currentframe += 1
		except VideoException as e:
			# close all resources
			close_all_resources(
				face_detector=face_detector,
				cam=cam,
				log_file=log_file,
				bar=bar,
				save_videos=save_videos,
				save_log=save_log
			)
			# raise it again
			raise e
		except Exception as e:
			print_log(e.args, log_file)
			print_log(logging.error(logging.traceback.format_exc()), log_file)
			video_exception = FaceDetector.update_exceptions()
			# if the maximum exceptions occurred during execution:
			if video_exception:
				# close all resources
				close_all_resources(
					face_detector=face_detector,
					cam=cam,
					log_file=log_file,
					bar=bar,
					save_videos=save_videos,
					save_log=save_log
				)

				# it removes the added exceptions as they will count as `VideoException`
				FaceDetector.quant_fired_exceptions -= FaceDetector.max_quant_exceptions
				
				# raise the `VideoException`
				raise VideoException(f'`FaceDetector` sub-class `detect_features` method exceeded the maximum number of times ({FaceDetector.max_quant_exceptions}) it can fail: {e}')
		finally:
			if save_log:
				if log_file.closed == False:
					log_file.write(f'Video: {out_path.split(os.path.sep)[-1]}, Status: {status}, Frame: {currentframe}, Num faces: {i + 1}, Distance: {euc_dist}\n')

			if realtime and ret != False:
				cv2.imshow('Face', frame)
				# Press Q on keyboard to exit
				if cv2.waitKey(25) & 0xFF == ord('q'):
					print_log('The video was terminated by the user', log_file)
					break
			
			if save_videos:
				if bar is None:
					bar = Bar('Processing...', max=int(cam.get(cv2.CAP_PROP_FRAME_COUNT)) + 1, check_tty=False, hide_cursor=False, suffix='%(index)d/%(max)d - %(eta)ds')

				bar.next()

	print_log(f'\nNumber of exceptions: {FaceDetector.quant_fired_exceptions}', log_file)
	print_log('\nThe video is over', log_file)

	# close all resources
	close_all_resources(
		face_detector=face_detector,
		cam=cam,
		log_file=log_file,
		bar=bar,
		save_videos=save_videos,
		save_log=save_log
	)

	# condition for using the video stabilizer
	if use_video_stabilizer:
		for i in range(len(face_detector.faces)):
			for facial_landmark in face_detector.faces[i].facial_landmarks:
				if facial_landmark is not None:
					stabilizer = VidStab()
					stabilizer.stabilize(
						input_path=os.path.join(out_path, f'{facial_landmark.name}.{dataset.image_extension}'), 
						output_path=os.path.join(out_path, f'{facial_landmark.name}_vidstab.avi'), 
						output_fourcc='DIVX',
						border_size=-5,
						)
	
	return video_json

def process_dataset(dataset: Dataset, out_dataset_path: str, video_name_pattern='Frame%05d.pgm',
					detector_type='FaceMesh', use_magnification=False):
	"""Function that receives the folder structure to process each video with the `process_video` function, \
		thus producing the artificial dataset (each video with the cutted ROIs, stabilized, etc).

	Notes
	-----
	- A JSON structured by subject and internal folder will save the path to each video, as well as the path to the respective PPG.
	- The execution log will be saved inside the `out_dataset_path` folder as 'dataset_log.txt'.
	- All paths will be checked during the process. An error will be issued without harming other procedures.
	- Each video will receive the same processing parameters:
		- FaceMesh as facial detector;
		- Threshold as stabilizer:
			- Bottom face: replace method
			- Left eye: mean method
			- Right eye: mean method
			- Middle-face: mean method

	Keyword arguments
	-----------------
	dataset : Dataset
		The `dataset` object must belong to a class that enherit the class `Dataset` and must have implemented all its methods.
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
		Path of the output folder where you want to save the artificial dataset. It should not exist to not overwrite the previous dataset.

	video_name_pattern : str
		The `video_name_pattern` parameter must be a string with the following formatter or none: (%d). E.g.:
			- 'Frame%05d.pgm' which will be done: pattern % 0 (%05d says the sequence will be five digits \
				long, with the first ones filled with zeros: Frame00000.pgm, Frame00001.pgm, Frame00002.pgm, ...). 
	"""
	start_time = datetime.now()

	# `out_dataset_path` folder should not exist to not overwrite the previous dataset.
	assert os.path.isdir(out_dataset_path) == False, f"This is an overwrite protection. `out_dataset_path` already exists: {out_dataset_path}."
	assert issubclass(dataset.__class__, Dataset), "The `dataset` object must belong to a class that enherit the class `Dataset`"

	os.mkdir(out_dataset_path)

	log_file = open(os.path.join(out_dataset_path, "dataset_log.txt"), "w")

	# Printing logs and informations
	print_log(f"\nStart of execution: {start_time}", log_file)
	print_log(f"""\rArtificial dataset path: {out_dataset_path}
			\rVideo name pattern: {video_name_pattern}\n""", log_file)

	print_log("Videos that will be processed:", log_file)
	[print_log(path['video_path'], log_file) for path in dataset.paths]
	
	print_log("PPGs that will be processed:", log_file)
	[print_log(path['ppg_path'], log_file) for path in dataset.paths]
	
	dataset_json = {}
	video_exception_count = 0
	exceptions_count = 0

	# PROCESSING EACH VIDEO

	# for each video
	for path in dataset.paths:
		subject_folder_name = path['subject_folder_name']
		subdir_folder_name = path['subdir_folder_name']
		video_path = path['video_path']
		ppg_path = path['ppg_path']

		print_log(f'\n{"=" * 10} Video {subject_folder_name}_{subdir_folder_name} {"=" * 10}', log_file)
		out_path = os.path.join(out_dataset_path, subject_folder_name, subdir_folder_name)

		kwargs = {
			"video_path":           video_path, 
			"out_path":             out_path, 
			"detector_type":        detector_type,
			"save_log":             True,
			"save_videos":          True,
			"realtime":             False,
			"use_threshold":        True,
			"distortion_fixed":     False,
			"use_frame_stabilizer": False,
			"use_eyes_lock":        False,
			"use_video_stabilizer": False,
			"threshold_modes":      {'bottom_face': 'replace', 'right_eye': 'mean', 'left_eye': 'mean'}, # , 'middle_face': 'mean'
			"dataset":              dataset,
			"use_magnification":    use_magnification,
		}
		
		print_log(f"\nProcessing video {subject_folder_name}_{subdir_folder_name} with the following arguments:", log_file)
		[print_log(f'{key}: {value}', log_file) for key, value in kwargs.items()]
		
		try:
			assert check_video_pattern(video_path, video_name_pattern, log_file), f"The following `video_path` wasn't found: {video_path}"

			# the following function creates the artificial dataset folders, executes and saves the cuts
			video_json = process_video(**kwargs)

			# it saves the produced dictionary before other operations
			dataset_json['_'.join([subject_folder_name, subdir_folder_name])] = video_json

			assert os.path.exists(ppg_path), f"A file wasn't find with the `ppg_path` string: {ppg_path}"
			print_log("\nProcessing PPG ...", log_file)
			dataset.ppg_flatten(
				pulse_path=ppg_path,
				out_path=out_path
			)

			dataset_json['_'.join([subject_folder_name, subdir_folder_name])]['PPG'] = os.path.join(out_path, 'pulseOx.mat')
		
			# it writes the JSON of the dataset to disk
			with open(os.path.join(out_dataset_path, "dataset_info.json"), 'w') as dataset_json_file:
				json.dump(dataset_json, fp=dataset_json_file, indent=1)
		except VideoException as e:
			print_log(f"\nVideoException occurred while processing video {subject_folder_name}/{subdir_folder_name}: {e}", log_file)
			video_exception_count += 1
		except Exception as e:
			print_log(f"\nUnmapped Exception: \n{logging.traceback.format_exc()}", log_file)
			exceptions_count += 1
		
		exceptions_count += FaceDetector.quant_fired_exceptions
	# Printing logs and informations
	print_log(f"\nNumber of Exceptions: {exceptions_count}", log_file)
	print_log(f"\nNumber of VideoExceptions: {video_exception_count}", log_file)
	end_time = datetime.now()
	print_log(f"\nEnd of execution: {end_time}", log_file)

	# it retrives the duration
	duration = end_time - start_time
	# it divides by 3600 and will have (hours, seconds left)
	hours = divmod(duration.total_seconds(), 3600)
	# it divides the seconds left by 60 and will have (minutes, seconds)
	minutes = divmod(hours[1], 60)
	
	print_log(f"\nTotal runtime: {hours[0]} hours and {minutes[0]} minutes", log_file)

	log_file.close()

def parser():
	"""Creates the ArgumentParser to get the user options."""
	parser = argparse.ArgumentParser(description="Dataset Creator")
	parser.add_argument("--base_dataset_path", type=str, required=True, help="Path to the base dataset folder (default: None)", default=None)
	parser.add_argument("--dataset_name", type=str, help="Name of the dataset you want to map (default: UBFC)", default='UBFC')
	parser.add_argument("--out_path", type=str, required=True, help="Output folder path (default: None)", default=None)
	parser.add_argument("--detector_type", type=str, help="Facial detector (default: 'FaceMesh')", default='FaceMesh')
	parser.add_argument("--image_extension", type=str.lower, help="Image format that should be used when saving artificial images on disk (default: 'png')", default='png')
	parser.add_argument("--use_magnification", action="store_true", help="To use video magnification? (default: False)", default=False)
	parser.add_argument("--camera_type", type=str.lower, help="In the MAHNOB dataset, use RGB or black and white videos? (default: 'color')", default='color')
	parser.add_argument("--only_emotions", action="store_true", help="In the MAHNOB dataset, filter the videos that have emotions? (default: False)", default=False)
	parser.add_argument("--video_name_pattern", type=str, help="String with the video's filename pattern (default: '')", default='')
	parser.add_argument("--subdir_name", nargs='+', help="List of folders that are inside (and must be processed) each subject's folder (default: [])", default=[])

	return parser.parse_args()

if __name__ == '__main__':
	
	opt = parser()

	print(json.dumps(opt.__dict__, indent=4))

	assert opt.dataset_name in dir(dataset_classes), f"The choosen `dataset_name` ({opt.dataset_name}) is not inside the `dataset_classes` module"
	dataset_cls = getattr(dataset_classes, opt.dataset_name)
	assert issubclass(dataset_cls, Dataset), f"The choosen `dataset_name` doesn\'t enhirets from `dataset_classes.Dataset`"
	assert dataset_cls is not Dataset, f"The choosen `dataset_name` shouldn\'t be 'Dataset'"

	init_params = {}
	map_dataset_params = {}

	if opt.dataset_name == 'MAHNOB':
		map_dataset_params['camera_type'] = opt.camera_type
		init_params['only_emotions'] = opt.only_emotions
	else:
		map_dataset_params['video_name_pattern'] = opt.video_name_pattern
		if opt.dataset_name == 'MRNirp':
			map_dataset_params['subdir_name'] = opt.subdir_name

	dataset = dataset_cls(image_extension=opt.image_extension, **init_params)

	dataset.map_dataset(
		base_dataset_path=opt.base_dataset_path, 
		**map_dataset_params
	)

	process_dataset(
	  	dataset=dataset,
	  	out_dataset_path=opt.out_path,
	  	video_name_pattern=opt.video_name_pattern,
		detector_type=opt.detector_type,
		use_magnification=opt.use_magnification
	)