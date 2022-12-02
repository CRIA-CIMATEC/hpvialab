<h3 align="center">Dataset Creator</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
</div>

---

<p align="center"> Creator of artificial dataset for clipping of videos and treatment of heart beats.
		<br> 
</p>

## 📝 Contents table

- [About](#About)
- [Getting Started](#getting_started)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [TODO](#TODO)
- [Flow of the algorithm using Threshold](#operation)

## 🧐 About <a name = "About"></a>

The set of functions and classes present in this folder refer to I / 0 implementations of videos, facial detectors, video stabilizers, distortions applied and viewing the results generated.

## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will provide a copy of the project running on your local machine for development, testing and usage purposes.

Folders / files and descriptions:
```
dataset_analyzer/               : Folder that contains set of scripts related to the analysis of the heartbeat dataset
	analyzer.ipynb          : Notebook that contains cells that call the functions needed to create the HTML of datasets analyzes
	ppg_analyzer.py         : Script that contains set of functions used to interpolate, treat and plot the BPMs and its measurements
	exploration_vX.html     : HTMLS containing analyzes related to video and heart rate dataset

merl_car_downloader/            : Folder that contains set of scripts related to a download form of datasets that are in google drive
	dataset_downloader.py   : Script that contains set of functions required to read a CSV and download a dataset that is on google drive (need to have the CSV to download)
	dataset_unzip.py        : Script that contains a small excerpt of code used to unzip zips larger than 1 GB

dataset_classes.py              : Script that contains set of classes responsible for abstract a Dataset class and implement its structure and specificities. (contains Super and Sub classes of Dataset, like MRNirp and UBFC)
dataset_creator.py              : Script that contains set of functions responsible for creating a general execution pipeline (entire dataset) and specialized (a video only).
dataset_creator.yml             : File that contains the requeried packages to run the dataset_creator pipeline.
biohmd.py                		: Script that contains the Tkinter class responsible for create a form to run the demo. It also contains the class representing the backend of the demo.
face_detector.py                : Script that contains the class responsible for determining the facial detector that will be used and their respective methods
face_utils.py                   : Script that contains set of classes responsible for abstracting the face and the detected facial landmarks, as well as contain methods capable of stabilizing and calculating data relating to user-detected face
mesh_points.py                  : Script containing set of points related to facial landmarks for the Facemesh detector
realtime_hmd_simulator.ipynb    : Notebook that contains cells that call the functions required to perform the tests of the dataset creation scripts
utils.py                        : Script that contains set of generic functions that are not specific to other files
video_comparison.py             : Script that contains set of functions used to create a simultaneous video viewer
```

## Prerequisites <a name = "prerequisites"></a>

What you need to run the scripts and how to install them.

```
python        : version 3.6.10
zipfile38     : version 0.0.3
numpy         : version 1.17.0
pandas        : version 1.1.3
natsort       : version 7.0.1
heartpy       : version 1.2.7
scipy         : version 1.4.1
matplotlib    : version 3.1.2
opencv-python : version 3.4.15.55
tkinter       : version 8.6.8
python-vlc    : version 3.0.12118
mediapipe     : version 0.8.0
vidstab       : version 1.7.3
imutils       : version 0.5.4
progress      : version 1.5
pyedflib      : version 0.1.17
pillow		  : version 7.0.0
cmake		  : version 3.21.3
dlib		  : version 19.21.1
tensorflow    : version 1.3.0 or 1.13.1
setproctitle  : version 1.1.10
configobj     : version 5.0.6
```

#### Installing Prerequisites

If you want to install the packages one by one:

```
conda create --name dataset_creator
conda activate dataset_creator
conda install python=3.6.10
pip install zipfile38==0.0.3
conda install numpy=1.17
conda install pandas=1.1.3
conda install natsort=7.0.1
pip install heartpy==1.2.7
conda install scipy=1.4.1
conda install matplotlib=3.1.2
pip install opencv-python==3.4.15.55
conda install tk==8.6.8
pip install python-vlc==3.0.12118
pip install mediapipe==0.8.0
pip install vidstab==1.7.3
pip install imutils==0.5.4
conda install progress=1.5
pip install pyedflib==0.1.17
pip install pillow==7.0.0
pip install cmake==3.21.3 --user
pip install dlib==19.21.1 --user
pip install tensorflow-gpu==1.3.0
pip install tensorflow==1.3.0 # If you have a GPU
pip install tensorflow-gpu==1.13.1 # IF VERSION 1.3.0 DOES NOT WORK
pip install tensorflow==1.13.1 # If you have a GPU
pip install setproctitle==1.1.10
pip install configobj==5.0.6
```

If you want to install the packages via ".yml" file:

```
conda env create -f dataset_creator.yml
conda activate dataset_creator
```

### Attention! 

~~1º There is an incompatibility between the `Heartpy` library and the `MediaPipe` library, as they require different versions of the `Numpy`, they are, respectively: numpy<=1.17 and numpy==1.19.3.
To avoid conflicts, they need to install them in different environments. In the case of this documentation, the `dataset_creator.yml` file contains all the packets except for `HeartPy` and `DLIB`.~~

~~2º To obtain the BPM of the MR NIRP dataset subjects must run the `ppg_analyser.explore_ppg` function separately from the artificial dataset generation code (and in the environment that the `Heartpy` library is installed), since there is incompatibility between the `Heartpy` library and the `Mediapipe`.~~

3º `DLIB` is not in the list of `dataset_creator.yml` packets because your installation needs the `cmake` on the machine, which will not always be possible to have. Because of this, the package will only be installed if the user wants (in case you want to run `SVM` as a facial detector). You can manually install the `DLIB` package with:
 - pip install cmake==3.21.3 --user
 - pip install dlib==19.21.1 --user

4º After resolving the package conflicts, the following changes took place:
 - numpy: from version 1.19.3 to 1.17.0;
 - heartpy: version 1.2.7 was kept;
 - mediapipe: from version 0.8.3 to 0.8.0;
 - opencv-python: from version 4.5.1.48 to 3.4.15.55.

5º If the VLC does not work and bid exception "no function 'libvlc_new'", run the following commands at the Linux terminal:
 - apt list --installed | grep vlc (Should not appear anything for now)
 - sudo apt-get install vlc
 - apt list --installed | grep vlc (Now the VLC and LIBVLC should now appear)

Stay tuned to package versions to avoid compatibility issues.

## 🚀 Usage <a name="usage"></a>

Run on entire dataset (To map another dataset, create another class that inherits and implements `dataset_classes.Dataset` to map another dataset):
```
from dataset_creator import process_dataset
from dataset_classes import MRNirp, UBFC

dataset = MRNirp()

dataset.map_dataset(
	base_dataset_path='/path/to/dataset/dataset_in', 
	subdir_name=['NIR'], 
	video_name_pattern='Frame%05d.pgm'
)

process_dataset(
	dataset=dataset,
	out_dataset_path='/path/to/dataset/dataset_out', 
	video_name_pattern='Frame%05d.pgm'
)

dataset = UBFC()

dataset.map_dataset(
	base_dataset_path='/path/to/dataset/dataset_in', 
	video_name_pattern='vid.avi'
)

process_dataset(
	dataset=dataset,
	out_dataset_path='/path/to/dataset/dataset_out', 
	video_name_pattern='vid.avi'
)
```

Run in a single video:
```
from dataset_creator import process_video
from dataset_classes import MRNirp, UBFC
import json
import os

dataset = MRNirp() # or UBFC()

kwargs = {
	"video_path":           '/path/to/dataset/video_in/Frame%05d.pgm', 
	"out_path":             '/path/to/dataset/dataset_out', 
	"detector_type":        'FaceMesh',
	"save_log":             True,
	"save_videos":          True,
	"realtime":             False,
	"use_threshold":        True,
	"distortion_fixed":     False,
	"use_frame_stabilizer": False,
	"use_eyes_lock":        False,
	"use_video_stabilizer": False,
	"threshold_modes":      {'bottom_face': 'replace', 'right_eye': 'mean', 'left_eye': 'mean'},  # , 'middle_face': 'mean'
	"dataset":              dataset,
}

video_json = process_video(**kwargs)

# Write Dataset JSON on the disc
with open(os.path.join(out_dataset_path, "dataset_info.json"), 'w') as dataset_json_file:
	json.dump(dataset_json, fp=dataset_json_file, indent=1)
```

Run the simultaneous viewer of videos (graphical interface):
```
python video_comparison.py
```

Run demo (terminal/command-line):
```
python biohmd.py --video_path demo_input_example/subject1_rgb_20s.avi --gt_path demo_input_example/pulseOx.mat --out_path dataset_out_pipeline --detector_type FaceMesh --use_threshold --neural_net_name meta-rppg --image_extension png --video_fps 30 --ppg_fps 30 --bottom_face_thr_type replace --right_eye_thr_type mean --left_eye_thr_type mean --location Local --location_path ../Meta_rPPG/checkpoints/meta_rPPG2_v3/latest_meta_rPPG2_v3.pth
```

```
python biohmd.py --input_type dataset --dataset_path /home/desafio01/Documents/Codes/bio_hmd/UBFC_DATASET/ubfc_raw_dataset/ --out_path /home/desafio01/Documents/Codes/bio_hmd/UBFC_DATASET/dataset_out_pipeline --detector_type FaceMesh --use_threshold --neural_net_name meta-rppg --image_extension png --video_fps 30 --ppg_fps 30 --bottom_face_thr_type replace --right_eye_thr_type mean --left_eye_thr_type mean --location Local --location_path ../Meta_rPPG/checkpoints/test_pretrain_2/latest_test_pretrain_2.pth --run_dataset --run_gt --run_pred --run_render --run_plot_render
```

Plot multivalorated fields of the MR-NIRP dataset:
```
from ppg_analyzer import plot_multivalued

plot_multivalued(
	ppg_path='/path/to/dataset/dataset_in/pulseOx.mat',
	subject_name='sujeito_1'
)
```

Plot graphics for PPGs and process BPMs according to PPG:
```
from ppg_analyzer import explore_ppg

explore_ppg(
	dataset_path='/path/to/dataset/dataset_in/',
	ppg_fps=60,
	segment_width=4,
	strides=[4],
	save_mat=False,
	save_plots=True,
	show=True
)
```

<!-- ### 🔧 Running tests <a name = "tests"></a>

Explain how to run the automated tests for this system.

#### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

#### And coding style tests

Explain what these tests test and why

```
Give an example
``` -->

## TODO <a name = "TODO"></a>

- [x] Translate Dataset Creator pipeline documentation to english
- [x] Translate Demo and Video Comparison documentation to english
- [x] Test open a sequence of images with `video_comparison.py`
- [x] The function `datasets_classes.MR_Nirp.ppg_flatten` must process automatically the BPMs according to the ppg when the `dataset_creator.process_dataset` is called (so you do not need the user to change environment and run `ppg_analyser.explore_ppg`)

##### video_comparison.py
- [ ] ~~Check the question of the size of the canvas in the frame~~
- [x] Add video controls near the progress bar
- [x] Customize progress bar
- [x] Check possibility to change the time of the progress bar to skip to another part of the video
- [ ] ~~Add option to remove and reorganize videos~~
- [ ] ~~Start `InitialDir` with the latest folder string accessed in the Choose Files window~~
- [ ] ~~Check if it is possible to use `askopenfilenames` to open multiple videos~~
- [ ] ~~Add log for debug from the tasks that the program is running~~

## Flow of the algorithm using Threshold <a name = "operation"></a>

1. The `process_video` function of the `dataset_creator.py` file receives the path of the video and the output path, as well as the options to choose the stabilizer and the detector;

2. The video is loaded with the class`VideoCapture` from `OpenCV`; 

3. Each frame read is passed to the method using `Facemesh` or `SVM` to detect the ROIs; 

4. In the region of the mouth, the frame is cut in a region larger than the final (temporary); 

5. In the cutting of the mouth region is applied the "barrel" distortion that uses the `Remap` method of `OpenCV`; 

6. Each of the ROIs is cropped from the frame (eyes) or the temporary region (mouth); 

7. Each ROI image passes through the `array` method of the `Numpy` library; 

8. Each ROI image passes through the `Resize` method of `OpenCV` with the final size being 400x400 (mouth ROI) and 100x100 (eyes ROIs);

9. The ROI image is converted from BGR to Gray using the `cvtcolor` method of `OpenCV`;

10. Each ROI image is written on the disc using the `imwrite` method of `OpenCV`;

11. The names of the written images on the disc receive the index of the position that are relative to the video;

12. The extension of all images written on disk is '.pgm' (Portable Gray Map) ('pgm' to infrared images and 'png' to RGB ones).