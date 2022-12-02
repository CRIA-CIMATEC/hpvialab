import numpy as np
import os
import json
import scipy.io
from cv2 import cv2
from settings import TrainOptions

opt = TrainOptions().get_options()


# opt.feature_image_path = '../../../Datasets/artificial_v1/dataset_info.json'
out_path = '../MetaRPPg_dataset_test'

if not os.path.isdir(out_path):
	os.mkdir(out_path)

f = open(opt.feature_image_path)
json_file = json.load(f)
f.close()

# arb_val = 1500
# max_frames = arb_val + opt.win_size * opt.fewshots

removed_keys = []

# for subname, sub in json_file.items():
  # length = len(sub['bottom_face']) 
  # if length < max_frames:
	# removed_keys.append(subname)
	# print(f'{subname} removido por não conter uma quantidade de frames superior a {max_frames}')

# for key in removed_keys:
	# json_file.pop(key)


dataset_json = {} 
dataset_info_json_path = os.path.join(out_path, 'dataset_info.json')

for name_subject, subject in json_file.items():
	dataset_json[name_subject] = {'ppg': [], 'image': [], 'mask': []}
	# dataset_json[name_subject]['ppg'] = scipy.io.loadmat(subject['PPG'])['pulseOxRecord'][0][:max_frames].tolist()
	dataset_json[name_subject]['ppg'] = scipy.io.loadmat(subject['PPG'])['pulseOxRecord'][0].tolist()
	subject_path = os.path.join(out_path, name_subject)
	
	frame_path = os.path.join(subject_path, 'frame')
	mask_path = os.path.join(subject_path, 'mask')
	
	if not os.path.isdir(subject_path):
		os.mkdir(subject_path)
	
	if not os.path.isdir(frame_path):
		os.mkdir(frame_path)
		
	if not os.path.isdir(mask_path):
		os.mkdir(mask_path)

	
	print(f'Diretório de gravação de frames: {frame_path}')
	print(f'Diretório de gravação de frames: {mask_path}')
	subject_frames = []
	subject_mask = []
	print(f"Agora em {name_subject}")
	# for i in range(len(subject['middle_face'][ : max_frames])):
	for i in range(len(subject['bottom_face'][ :5300])): 
		frame = cv2.imread(subject['bottom_face'][i])
		# print(subject['bottom_face'][i])
		# print(f"frame: {frame}")
		frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
		left_eye = cv2.imread(subject['left_eye'][i])
		right_eye = cv2.imread(subject['right_eye'][i])
		eyes = cv2.hconcat([right_eye, left_eye])
		frame = cv2.vconcat([eyes, frame])
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = frame[:-100, :]
		# h, w, c = frame.shape
		frame = cv2.resize(frame, (0,0), fx=0.32, fy=0.32)
		# frame = np.transpose(frame, (2, 0, 1))
		mask = np.ones_like(frame) * 255
		frame_name = os.path.join(frame_path, f'{i}.pgm')
		mask_name = os.path.join(mask_path, f'{i}.pgm')
		cv2.imwrite(frame_name, frame)
		cv2.imwrite(mask_name, mask)
		subject_frames.append(frame_name)
		subject_mask.append(mask_name)
	dataset_json[name_subject]['image'] = subject_frames
	dataset_json[name_subject]['mask'] = subject_mask

print(dataset_json)

with open(dataset_info_json_path, 'w') as outfile:
  json.dump(dataset_json, outfile, indent = 1)
	
