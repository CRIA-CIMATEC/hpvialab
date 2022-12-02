# Import packages and libraries
import logging
import os
import tkinter.constants as const
import tkinter as tk
from tkinter import ttk
import json
from tkinter.filedialog import askdirectory, askopenfilename
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
import neurokit2

import numpy as np
from dataset_classes import DemoBasicDataset, MRNirp
from utils import create_features_video, make_dir
from dataset_creator import process_video
from dataset_analyzer.ppg_analyzer import ppg_to_bpm
import argparse
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 6})
plt.rc('legend', fontsize=6)

import sys
sys.path.append('..')
from EVM.test import run_test as evm_test, TestOptions
from EVM.eval import run_eval as evm_eval, EvalOptions
from Meta_rPPG.eval import run_eval as meta_eval, TrainOptions
from Meta_rPPG.test import run_test as meta_test, TrainOptions
from postprocessing.utils import postprocessing

'''This file is to mount the BIOHMD, collect the paths and parameters to run the prediction'''

class DemoForm(tk.Toplevel):
	"""Class responsible to create the main form with the demo params."""
	def __init__(self, parent, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		# Size of the window
		# self.geometry('800x300')
		# Sets the window size
		screenwidth = parent.winfo_screenwidth()
		screenheight = parent.winfo_screenheight()
		alignstr = '%dx%d' % (screenwidth, screenheight)
		self.geometry(alignstr)

		self.kwargs = None
		self.parent = parent

		# FORM VARIABLES
		self.use_thr = tk.BooleanVar(value=1)
		self.use_vidstab_frame = tk.BooleanVar()
		self.use_vidstab_full = tk.BooleanVar()
		self.message_video_fps = tk.StringVar('')
		self.message_ppg = tk.StringVar('')
		self.message_img = tk.StringVar('')
		self.message_out = tk.StringVar('')
		self.message_video = tk.StringVar('')
		self.message_groud_truth = tk.StringVar('')
		self.message_pth = tk.StringVar('')
		self.message_left_eye = tk.StringVar('')
		self.message_right_eye = tk.StringVar('')
		self.message_bottom_face = tk.StringVar('')
	   
		self.input_type = tk.StringVar()
		self.video_path = tk.StringVar()
		self.bottom_face_path = tk.StringVar()
		self.right_eye_path = tk.StringVar()
		self.left_eye_path = tk.StringVar()
		self.ground_truth = tk.StringVar()
		self.out_path = tk.StringVar()
		self.detector = tk.StringVar()
		self.network = tk.StringVar()
		self.location = tk.StringVar()
		self.location_entry = tk.StringVar()

		self.image_ext_entry = tk.StringVar(value='png')
		self.video_fps_entry = tk.IntVar(value=30)
		self.ppg_fps_entry = tk.IntVar(value=30)
		self.bottom_face_thr_type = tk.StringVar(value='replace')
		self.right_eye_thr_type = tk.StringVar(value='mean')
		self.left_eye_thr_type = tk.StringVar(value='mean')
		self.use_magnification = tk.BooleanVar(value=False)
		self.use_postprocessing = tk.BooleanVar(value=False)
		
		self.run_dataset = tk.BooleanVar(value=1)
		self.run_gt = tk.BooleanVar(value=1)
		self.run_pred = tk.BooleanVar(value=1)
		self.run_render = tk.BooleanVar(value=1)
		self.run_plot_render = tk.BooleanVar(value=1)

		self.all = tk.BooleanVar(value=1)
		self.none = tk.BooleanVar(value=0)
		self.resume_entry = tk.StringVar(value=self.out_path.get())

		self.location_label = tk.StringVar()
		self.location_label_options = ['NN weight path:', 'API\'s URL:']
		self.location_label.set(self.location_label_options[0])

		# FORM LISTS AND TEXTS
		self.detector_list = ["FaceMesh", "SVM"]
		self.net_list = ["Meta-rPPG", "EVM"]
		self.location_list = ["Local", "Server"]
		
		self.input_types = ['Entire_face', 'ROIs']
		self.rois = ['Bottom face', 'Right eye', 'Left eye']
		self.rois_comboboxes = []
		self.thr_types = ['mean', 'replace']

		self.str_bt_search = 'Search'
		self.str_bt_process = 'Process Video'
		self.str_bt_reload = 'Reload options'

		self.str_lb_input_type = 'Input type:'
		self.str_lb_video_path = 'Video file\'s path:'
		self.str_lb_bottom_path = 'Bottom face file\'s path:'
		self.str_lb_right_path = 'Right eye file\'s path:'
		self.str_lb_left_path = 'Left eye file\'s path:'
		self.str_lb_gt_path = 'Ground truth path:'
		self.str_lb_out_path = 'Output folder path:'
		self.str_lb_detector = 'Facial detector:'
		self.str_lb_stab = 'Stabilizers:'
		self.str_lb_thr = 'Threshold'
		self.str_lb_frame_stab = 'VidStab - Frame by frame'
		self.str_lb_full_stab = 'VidStab - Entire Video'
		self.str_lb_network = 'Neural network (NN):'
		self.str_lb_location = 'Where to process:'

		self.str_lb_image_ext = 'Image extension:'

		self.str_lb_video_fps = 'Video FPS:'
		self.str_lb_ppg_fps = 'PPG FPS:'
		self.str_lb_thr_types = 'Threshold types:'
		self.str_lb_bottom_face_thr = 'Bottom face:'
		self.str_lb_right_eye_thr = 'Right eye:'
		self.str_lb_left_eye_thr = 'Left eye:'
		self.str_lb_use_magnification = 'Video Magnification:'
		self.str_lb_use_postprocessing = 'Post processing:'

		self.str_lb_resume = 'Resume path:'
		self.str_lb_re_run = 'Re-run:'
		self.str_lb_all = 'All'
		self.str_lb_none = 'None'
		self.str_lb_run_dataset = 'Video preprocessing'
		self.str_lb_run_gt = 'Ground truth preprocessing'
		self.str_lb_run_pred = 'Neural network prediction'
		self.str_lb_run_plot_render = 'Render plot'
		self.str_lb_run_render = 'Render ROIs'

		self.title(self.str_bt_process)

		notebook = ttk.Notebook(self)

		# builds tabs inside the notebook
		self.build_io_tab(notebook)
		self.build_preprocessing_tab(notebook)
		self.build_net_params_tab(notebook)
		self.build_resume_tab(notebook)

		notebook.pack(expand=True, fill='both', side='top')

		# Adds an event to show/hide the reload button
		notebook.bind("<<NotebookTabChanged>>", lambda event: self.notebook_tab_changed())
		# buttons reload and run
		tk.Button(self, text=self.str_bt_reload, command=self.reload_options, name='btn_reload_options').pack(expand=False, fill='x', side='left')
		tk.Button(self, text=self.str_bt_process, command=self.run_demo).pack(expand=False, fill='x', side='right')

	def build_io_tab(self, notebook: ttk.Notebook) -> None:
		"""Method that creates the 'I/O' tab of the `notebook` parameter.
	
		Keyword arguments
		-----------------
		notebook : ttk.Notebook
		Notebook object that will be the tab's parent.
		"""
		tab_io = ttk.Frame(notebook, name='io')
		tab_io.pack(expand=True, fill='both')

		# Configures the third column to be wider than the other
		tab_io.grid_columnconfigure(2, weight=1)
		row_idx = 0

		# INPUT TYPE
		tk.Label(tab_io, text=self.str_lb_input_type).grid(column=0, row=row_idx)
		combo_input_types = ttk.Combobox(tab_io, state="readonly", values=self.input_types, textvariable=self.input_type)
		combo_input_types.set(self.input_types[0])
		combo_input_types.grid(column=2, row=row_idx, sticky='ew')

		# VIDEO PATH
		row_idx += 1
		tk.Label(tab_io, text=self.str_lb_video_path, name='entire_face_item_0').grid(column=0, row=row_idx)
		tk.Button(tab_io, text=self.str_bt_search, command=lambda: self.select_video_form(self.video_path), name='entire_face_item_1').grid(column=1, row=row_idx)
		video_path_entry = tk.Label(tab_io, textvariable=self.video_path, name='entire_face_item_2')
		video_path_entry.grid(column=2, row=row_idx, sticky='ew')

		# BOTTOM FACE PATH
		row_idx += 1
		tk.Label(tab_io, text=self.str_lb_bottom_path, name='roi_item_0').grid(column=0, row=row_idx) 
		tk.Button(tab_io, text=self.str_bt_search, command=lambda: self.select_video_form(self.bottom_face_path), name='roi_item_1').grid(column=1, row=row_idx)
		bottom_face_path = tk.Label(tab_io, textvariable=self.bottom_face_path, name='roi_item_2')
		bottom_face_path.grid(column=2, row=row_idx, sticky='ew')

		# RIGHT EYE PATH
		row_idx += 1
		tk.Label(tab_io, text=self.str_lb_right_path, name='roi_item_3').grid(column=0, row=row_idx)
		tk.Button(tab_io, text=self.str_bt_search, command=lambda: self.select_video_form(self.right_eye_path), name='roi_item_4').grid(column=1, row=row_idx)
		right_eye_path = tk.Label(tab_io, textvariable=self.right_eye_path, name='roi_item_5')
		right_eye_path.grid(column=2, row=row_idx, sticky='ew')

		# LEFT EYE PATH
		row_idx += 1
		tk.Label(tab_io, text=self.str_lb_left_path, name='roi_item_6').grid(column=0, row=row_idx)
		tk.Button(tab_io, text=self.str_bt_search, command=lambda: self.select_video_form(self.left_eye_path), name='roi_item_7').grid(column=1, row=row_idx)
		left_eye_path = tk.Label(tab_io, textvariable=self.left_eye_path, name='roi_item_8')
		left_eye_path.grid(column=2, row=row_idx, sticky='ew')

		# adds an event to the input type Combobox update the path entries of the form
		combo_input_types.bind(
			'<<ComboboxSelected>>', 
			lambda event, entry=combo_input_types: self.change_input_entries(entry.get())
		)

		self.change_input_entries(combo_input_types.get())

		# GROUND TRUTH PATH
		row_idx += 1
		tk.Label(tab_io, text=self.str_lb_gt_path).grid(column=0, row=row_idx)
		tk.Button(tab_io, text=self.str_bt_search, command=self.select_gt_form).grid(column=1, row=row_idx)
		tk.Button(tab_io, text='No ground truth', command=self.select_no_gt_form).grid(column=3, row=row_idx)
		ground_truth = tk.Label(tab_io, textvariable=self.ground_truth)
		ground_truth.grid(column=2, row=row_idx, sticky='ew')

		# OUT PATH
		row_idx += 1
		tk.Label(tab_io, text=self.str_lb_out_path).grid(column=0, row=row_idx)
		tk.Button(tab_io, text=self.str_bt_search, command=lambda: self.select_dir_form(self.out_path)).grid(column=1, row=row_idx)
		out_path = tk.Label(tab_io, textvariable=self.out_path)
		out_path.grid(column=2, row=row_idx, sticky='ew')

		# Adds a pad between each line of this tab
		[tab_io.grid_rowconfigure(linha, pad=10) for linha in range(tab_io.grid_size()[1])]
		notebook.add(tab_io, text='I/O')

	def build_preprocessing_tab(self, notebook: ttk.Notebook) -> None:
		"""Method that creates the 'Precessing' tab of the `notebook` parameter.

		Keyword arguments
		-----------------
		notebook : ttk.Notebook
			Notebook object that will be the tab's parent.
		"""
		# Creating frame  
		tab_preprocessing = ttk.Frame(notebook, name='preprocessing')
		tab_preprocessing.pack(expand=True, fill='both')

		# Configures the third column to be wider than the other
		tab_preprocessing.grid_columnconfigure(2, weight=1)
		row_idx = 0

		# DETECTOR TYPE
		row_idx += 1
		tk.Label(tab_preprocessing, text=self.str_lb_detector).grid(column=0, row=row_idx)
		combo_detector = ttk.Combobox(tab_preprocessing, state="readonly", values=self.detector_list, textvariable=self.detector)
		combo_detector.set(self.detector_list[0])
		combo_detector.grid(column=2, row=row_idx, sticky='ew')

		# STABILIZERS
		row_idx += 1
		tk.Label(tab_preprocessing, text=self.str_lb_stab).grid(column=0, row=row_idx)
		thr_check = tk.Checkbutton(tab_preprocessing, text=self.str_lb_thr, justify=const.LEFT, variable=self.use_thr, command=self.change_types_visibility)
		thr_check.grid(column=2, row=row_idx, sticky='w')

		# Check buttons to decide the use of Stabilizers
		row_idx += 1
		tk.Checkbutton(tab_preprocessing, text=self.str_lb_frame_stab, variable=self.use_vidstab_frame).grid(column=2, row=row_idx, sticky='w')

		row_idx += 1
		tk.Checkbutton(tab_preprocessing, text=self.str_lb_full_stab, variable=self.use_vidstab_full).grid(column=2, row=row_idx, sticky='w')

		# THRESHOLD TYPE
		row_idx += 1
		tk.Label(tab_preprocessing, text=self.str_lb_thr_types).grid(column=0, row=row_idx)

		# bottom_face
		row_idx += 1
		tk.Label(tab_preprocessing, text=self.str_lb_bottom_face_thr).grid(column=1, row=row_idx)
		combo_net = ttk.Combobox(tab_preprocessing, state="readonly", values = self.thr_types, textvariable=self.bottom_face_thr_type)
		combo_net.set(self.thr_types[1])
		combo_net.grid(column=2, row=row_idx, sticky='ew')
		self.rois_comboboxes.append(combo_net)

		# Left_eye
		row_idx += 1
		tk.Label(tab_preprocessing, text=self.str_lb_left_eye_thr).grid(column=1, row=row_idx)
		combo_net = ttk.Combobox(tab_preprocessing, state="readonly", values = self.thr_types, textvariable=self.left_eye_thr_type)
		combo_net.set(self.thr_types[0])
		combo_net.grid(column=2, row=row_idx, sticky='ew')
		self.rois_comboboxes.append(combo_net)

		# Right_eye
		row_idx += 1
		tk.Label(tab_preprocessing, text=self.str_lb_right_eye_thr).grid(column=1, row=row_idx)
		combo_net = ttk.Combobox(tab_preprocessing, state="readonly", values = self.thr_types, textvariable=self.right_eye_thr_type)
		combo_net.set(self.thr_types[0])
		combo_net.grid(column=2, row=row_idx, sticky='ew')
		self.rois_comboboxes.append(combo_net)

		# Adds a pad between each line of this tab
		[tab_preprocessing.grid_rowconfigure(linha, pad=10) for linha in range(tab_preprocessing.grid_size()[1])]
		notebook.add(tab_preprocessing, text='Preprocessing')

	def build_net_params_tab(self, notebook: ttk.Notebook) -> None:
		"""Method that creates the 'Network and Params' tab of the `notebook` parameter.
		
		Keyword arguments
		-----------------
		notebook : ttk.Notebook
			Notebook object that will be the tab's parent.
		"""

		# Creating frame
		tab_net_params = ttk.Frame(notebook, name='net_params')
		tab_net_params.pack(expand=True, fill='both')

		# Configures the third column to be wider than the other
		tab_net_params.grid_columnconfigure(2, weight=1)
		row_idx = 0

		# IMAGE EXTENSION
		row_idx += 1
		tk.Label(tab_net_params, text=self.str_lb_image_ext).grid(column=0, row=row_idx)
		image_ext_entry = tk.Entry(tab_net_params, textvariable=self.image_ext_entry) # Input for image extension
		tk.Label(tab_net_params, textvariable=self.message_img, fg='red', bd=0).grid(column=1, row=row_idx)
		image_ext_entry.grid(column=2, row=row_idx, sticky='ew')

		# configures the Entry scroll to show the end of the string
		image_ext_entry.xview('end')

		# VIDEO FPS
		row_idx += 1
		tk.Label(tab_net_params, text=self.str_lb_video_fps).grid(column=0, row=row_idx)
		video_fps_entry = tk.Entry(tab_net_params, textvariable=self.video_fps_entry) # Input of fps value
		tk.Label(tab_net_params, textvariable=self.message_video_fps, fg='red', bd=0).grid(column=1, row=row_idx)
		video_fps_entry.grid(column=2, row=row_idx, sticky='ew')
		video_fps_entry.xview('end')

		# PPG FPS
		row_idx += 1
		tk.Label(tab_net_params, text=self.str_lb_ppg_fps).grid(column=0, row=row_idx)
		ppg_fps_entry = tk.Entry(tab_net_params, textvariable=self.ppg_fps_entry) # Input of ppg fps value
		tk.Label(tab_net_params, textvariable=self.message_ppg, fg='red', bd=0).grid(column=1, row=row_idx)
		ppg_fps_entry.grid(column=2, row=row_idx, sticky='ew')
		ppg_fps_entry.xview('end')

		# VIDEO_MAGNIFICATION
		row_idx += 1
		tk.Label(tab_net_params, text=self.str_lb_use_magnification).grid(column=0, row=row_idx)
		# Checkbutton to decide to use video mag
		thr_check = tk.Checkbutton(tab_net_params, text=self.str_lb_use_magnification[:-1], justify=const.LEFT, variable=self.use_magnification) 
		thr_check.grid(column=2, row=row_idx, sticky='w')

		# POST_PROCESSING
		row_idx += 1
		tk.Label(tab_net_params, text=self.str_lb_use_postprocessing).grid(column=0, row=row_idx)
		# Checkbutton to decide to use post processing
		thr_check = tk.Checkbutton(tab_net_params, text=self.str_lb_use_postprocessing[:-1], justify=const.LEFT, variable=self.use_postprocessing) 
		thr_check.grid(column=2, row=row_idx, sticky='w')

		# NETWORK
		row_idx += 1
		tk.Label(tab_net_params, text=self.str_lb_network).grid(column=0, row=row_idx)
		combo_net = ttk.Combobox(tab_net_params, state="readonly", values = self.net_list, textvariable=self.network)
		combo_net.set(self.net_list[0])
		combo_net.grid(column=2, row=row_idx, sticky='ew')

		# WHERE TO PROCESS
		row_idx += 1
		tk.Label(tab_net_params, text=self.str_lb_location).grid(column=0, row=row_idx)
		combo_location = ttk.Combobox(tab_net_params, state="readonly", textvariable=self.location, values=self.location_list)
		combo_location.set(self.location_list[0])
		combo_location.grid(column=2, row=row_idx, sticky='ew')

		# URL or NET WEIGHT LOCATION
		row_idx += 1
		tk.Label(tab_net_params, textvariable=self.location_label).grid(column=0, row=row_idx)
		location_entry = tk.Label(tab_net_params, textvariable=self.location_entry)
		location_entry.grid(column=2, row=row_idx, sticky='ew')

		bt_net_path = tk.Button(tab_net_params, text=self.str_bt_search, command=self.select_weight_form)
		bt_net_path.grid(column=1, row=row_idx)
		tk.Label(tab_net_params, textvariable=self.message_pth, fg='red', bd=0).grid(column=1, row=row_idx+1)

		# event, entry=combo_location: self.change_location_ adds an event to the combo_location Combobox update (normal/disabled) the bt_net_path of the form
		# combo_location.bind('<<ComboboxSelected>>', lambda form(entry.get(), button=bt_net_path))

		# Adds a pad between each line of this tab
		[tab_net_params.grid_rowconfigure(linha, pad=10) for linha in range(tab_net_params.grid_size()[1])]
		notebook.add(tab_net_params, text='Network and Params', sticky=tk.NSEW)

		def callback_video_FPS(input):
			'''Method to check if the FPS value of video is a valid digit and if necessary, return an error message.'''

			# Checking FPS of video to return only digits
			while input.isdigit() or not input:
				self.message_video_fps.set(' ')
				return True
			else:
				# Returning an error message and blocking writing if the user tries to type a letter
				self.message_video_fps.set('The VIDEO FPS must be a number')
				return False

		reg_fps_video = self.register(callback_video_FPS)
		video_fps_entry.config(validate="key", validatecommand=(reg_fps_video, '%P'))

		def callback_PPG_FPS(input):
			'''Method to check if the FPS value of PPG is a valid digit and if necessary, return an error message.'''

			# Checking FPS of PPG to return only digits
			while input.isdigit() or not input:
				self.message_ppg.set(' ')
				return True
			else:
				# Returning an error message and blocking writing if the user tries to type a letter
				self.message_ppg.set('The PPG FPS must be a number')
				return False

		reg_fps_ppg = self.register(callback_PPG_FPS)
		ppg_fps_entry.config(validate="key", validatecommand=(reg_fps_ppg, '%P'))

		# Support extensions
		def callback_image_ext(input):
			'''Method to check the supported image extension and return a message'''
			# creating list with supported formats
			supported_image_formats = [
			'jpeg', 'jpg', 'jpe', 
			'bmp', 'dib', 'jp2', 
			'png', 'webp', 'pbm', 
			'pgm', 'ppm', 'pxm', 
			'pnm', 'pfm', 'sr', 
			'ras', 'tiff', 'tif', 
			'exr', 'hdr', 'pic']

			# turning input to lowercase for check extension
			input = input.lower()

			# Creating loop to check image format and return possible error messages
			while input in supported_image_formats:
				self.message_img.set(' ')
				return True
			else:
				# Returning error messages if you try to type just a number or if the image format is not among those supported	
				self.message_img.set('Incorrect image format')
				if input.isdigit() or input.isspace():
					return False
				else:
					self.message_img.set('Incorrect image format')
					return True               
		   
		reg_image = self.register(callback_image_ext)
		image_ext_entry.config(validate="key", validatecommand=(reg_image, '%P'))

	def build_resume_tab(self, notebook: ttk.Notebook) -> None:
		"""Method that creates the 'Resume' tab of the `notebook` parameter.
		
		Keyword arguments
		-----------------
		notebook : ttk.Notebook
			Notebook object that will be the tab's parent.
		"""
		tab_resume = ttk.Frame(notebook, name='resume')
		tab_resume.pack(expand=True, fill='both')
		
		# Configures the third column to be wider than the other
		tab_resume.grid_columnconfigure(2, weight=1)
		row_idx = 0

		# RESUME FOLDER
		row_idx += 1
		tk.Label(tab_resume, text=self.str_lb_resume).grid(column=0, row=row_idx)
		tk.Button(tab_resume, text=self.str_bt_search, command=lambda: self.select_dir_form(self.resume_entry)).grid(column=1, row=row_idx)
		resume_entry = tk.Label(tab_resume, textvariable=self.resume_entry)
		resume_entry.grid(column=2, row=row_idx, sticky='ew')
		# configures the Entry scroll to show the end of the string

		# RE-RUN OPTIONS

		# Checkbuttons and labels
		row_idx += 1 # row_idx variable used to define the vertical position of the element on the screen, being updated for each element created
		tk.Label(tab_resume, text=self.str_lb_re_run).grid(column=0, row=row_idx)
		tk.Checkbutton(tab_resume, text=self.str_lb_all, variable=self.all, command=self.turn_on_all_buttons).\
			grid(column=2, row=row_idx, sticky='w')

		row_idx += 1
		tk.Checkbutton(tab_resume, text=self.str_lb_none, variable=self.none, command=self.turn_off_all_buttons).\
			grid(column=2, row=row_idx, sticky='w')

		row_idx += 1
		tk.Checkbutton(tab_resume, text=self.str_lb_run_dataset, variable=self.run_dataset, command=self.change_form_state).\
			grid(column=2, row=row_idx, sticky='w')

		row_idx += 1
		tk.Checkbutton(tab_resume, text=self.str_lb_run_gt, variable=self.run_gt, command=self.change_form_state).\
			grid(column=2, row=row_idx, sticky='w')

		row_idx += 1
		tk.Checkbutton(tab_resume, text=self.str_lb_run_pred, variable=self.run_pred, command=self.change_form_state).\
			grid(column=2, row=row_idx, sticky='w')
		
		row_idx += 1
		tk.Checkbutton(tab_resume, text=self.str_lb_run_plot_render, variable=self.run_plot_render, command=self.change_form_state).\
			grid(column=2, row=row_idx, sticky='w')

		row_idx += 1
		tk.Checkbutton(tab_resume, text=self.str_lb_run_render, variable=self.run_render, command=self.change_form_state).\
			grid(column=2, row=row_idx, sticky='w')

		# Adds a pad between each line of this tab
		[tab_resume.grid_rowconfigure(linha, pad=10) for linha in range(2)]
		notebook.add(tab_resume, text='Resume')

	def change_types_visibility(self) -> None:
		"""Method that changes the state of the ROIs comboboxes based on the threshold CheckButton."""
		for comboboxes in self.rois_comboboxes:
			if self.use_thr.get():
				state = "enabled"
			else:
				state = "disabled"

			comboboxes["state"] = state

	def change_location_form(self, location: str, button) -> None:
		"""Method that changes the Search button state based on the chosen process location.
		
		Keyword arguments
		-----------------
		location : str
			Location that was chosen by the user. It could be 'Local' or 'Server'.
		button : tk.Button
			Search button that must have its state changed.
		"""
		if location.lower() == self.location_list[0].lower():
			self.location_label.set(self.location_label_options[0])
			button["state"] = "normal"
		elif location.lower() == self.location_list[1].lower():
			self.location_label.set(self.location_label_options[1])
			button["state"] = "disabled"
		else:
			print(f'The specified processing location has not been mapped: {location}\nMapped locations: {self.location_list}')

	def change_input_entries(self, input_type) -> None:
		"""Method that show/hide the paths of the form based on the `input_type`.
		
		Keyword arguments
		-----------------
		input_type : str
			Input type that will determine if the user is passing a ROI or a entire face to the demo.
		"""
		# gets the `io` tab of the notebook
		io = self.children['!notebook'].children['io']

		# if input_type == 'entire_face':
		if input_type.lower() == self.input_types[0].lower():
			for i in range(0, 9):
				# removes grid from each roi widget
				io.children[f'roi_item_{i}'].grid_remove()
			for i in range(0, 3):
				# grids each entire_face widget
				io.children[f'entire_face_item_{i}'].grid()
				# elif input_type == 'rois':
		elif input_type.lower() == self.input_types[1].lower():
			for i in range(0, 3):
				# removes grid from each entire_face widget
				io.children[f'entire_face_item_{i}'].grid_remove()
			for i in range(0, 9):
				# grids each roi widget
				io.children[f'roi_item_{i}'].grid()
		else:
			print(f'The specified video input type has not been mapped: {input_type}\nMapped input ypes: {self.input_types}')

		# updates the form size based on the `io` form size
		self.after(100, self.update_geometry, io)

	def update_geometry(self, frame) -> None:
		"""Method that updates the form size based on the `frame` form size.
		
		Keyword arguments
		-----------------
		frame : tk.Frame
			Frame object that has children at its grid.
		"""
		self.update()
		width, height = frame.grid_bbox()[2:]
		# adds 135 pixels to the `height` to show the 'Process Video' button
		self.geometry(f'{width}x{height*2}')

	def reload_options(self) -> None:
		"""Method that updates the form based on the 'demo_kwargs.json' 
		file when the user wants to reload older options."""
		kwargs_path = os.path.join(self.resume_entry.get(), 'demo_kwargs.json')
		if os.path.isfile(kwargs_path):
			with open(kwargs_path, 'r') as fp:
				kwargs = json.load(fp)
			self.input_type.set(kwargs.get('input_type', 'Entire_face'))
			self.change_input_entries(self.input_type.get())
			self.video_path.set(kwargs.get('video_path', ''))
			self.bottom_face_path.set(kwargs.get('bottom_face_path', ''))
			self.right_eye_path.set(kwargs.get('right_eye_path', ''))
			self.left_eye_path.set(kwargs.get('left_eye_path', ''))
			self.ground_truth.set(kwargs.get('gt_path', ''))
			self.out_path.set(kwargs.get('out_path', ''))
			self.detector.set(kwargs.get('detector_type', ''))
			self.use_thr.set(kwargs.get("use_threshold", True))
			self.change_types_visibility()
			self.use_vidstab_frame.set(kwargs.get("use_frame_stabilizer", False))
			self.use_vidstab_full.set(kwargs.get("use_video_stabilizer", False))
			self.network.set(kwargs.get('neural_net_name', ''))
			self.location.set(kwargs.get('location', ''))
			self.location_entry.set(kwargs.get('location_path', ''))

			self.image_ext_entry.set(kwargs.get('image_extension', ''))
			self.video_fps_entry.set(kwargs.get('video_fps', 30))
			self.ppg_fps_entry.set(kwargs.get('ppg_fps', 30))
			self.bottom_face_thr_type.set(kwargs.get('bottom_face_thr_type', ''))
			self.right_eye_thr_type.set(kwargs.get('right_eye_thr_type', ''))
			self.left_eye_thr_type.set(kwargs.get('left_eye_thr_type', ''))
			self.use_magnification.set(kwargs.get('use_magnification', False))
			self.use_postprocessing.set(kwargs.get('use_postprocessing', False))

	def run_demo(self) -> None:
		"""Method that sends all form to the `Demo` and gets its output code."""
		demo = Demo()
		# Gets
		demo.kwargs['input_type'] = self.input_type.get().lower()
		demo.kwargs['video_path'] = self.video_path.get()
		demo.kwargs['bottom_face_path'] = self.bottom_face_path.get()
		demo.kwargs['right_eye_path'] = self.right_eye_path.get()
		demo.kwargs['left_eye_path'] = self.left_eye_path.get()
		demo.kwargs['gt_path'] = self.ground_truth.get()
		demo.kwargs['out_path'] = self.out_path.get()
		demo.kwargs['detector_type'] = self.detector.get()
		demo.kwargs['use_threshold'] = self.use_thr.get()
		demo.kwargs['use_frame_stabilizer'] = self.use_vidstab_frame.get()
		demo.kwargs['use_video_stabilizer'] = self.use_vidstab_full.get()
		demo.kwargs['neural_net_name'] = self.network.get().lower()
		demo.kwargs['location'] = self.location.get()
		demo.kwargs['location_path'] = self.location_entry.get()

		demo.kwargs['image_extension'] = self.image_ext_entry.get().lower()
		demo.kwargs['video_fps'] = self.video_fps_entry.get()
		demo.kwargs['ppg_fps'] = self.ppg_fps_entry.get()
		demo.kwargs['bottom_face_thr_type'] = self.bottom_face_thr_type.get().lower()
		demo.kwargs['right_eye_thr_type'] = self.right_eye_thr_type.get().lower()
		demo.kwargs['left_eye_thr_type'] = self.left_eye_thr_type.get().lower()
		demo.kwargs['use_magnification'] = self.use_magnification.get()
		demo.kwargs['use_postprocessing'] = self.use_postprocessing.get()
		demo.kwargs['resume_folder'] = self.resume_entry.get()
		demo.kwargs['run_dataset'] = self.run_dataset.get()
		demo.kwargs['run_gt'] = self.run_gt.get()
		demo.kwargs['run_pred'] = self.run_pred.get()
		demo.kwargs['run_render'] = self.run_render.get()
		demo.kwargs['run_plot_render'] = self.run_plot_render.get()

		self.kwargs = demo.run()
		# printing to return run status
		print(f'STATUS CODE: {self.kwargs["status_code"]}')
		self.parent.show_results(**self.kwargs)
		# destroying the frame
		self.destroy()

	def select_video_form(self, text_variable: tk.StringVar) -> None:
		"""Method that ask a video filename to the user and updates the `text_variable` from an Entry.
		
		Keyword arguments
		-----------------
		text_variable : tk.StringVar
			Tkinter string variable that will be updated with the user input.
		"""
		filetypes=(
			("Audio Video Interleave", "*.avi"),
			("Matroska", "*.mkv"),
			("MP4", "*.mp4"),
		)
		file = askopenfilename(parent=self, filetypes=filetypes)
		if file:
			text_variable.set(file)

	def select_weight_form(self) -> None:
		"""Method that ask a weight filename to the user and updates the `location_entry` StringVar."""
		filetypes=(
			('', "*.pth"),
		)
		file = askopenfilename(parent=self, filetypes=filetypes)
		if file:
			self.location_entry.set(file)

	def select_dir_form(self, text_variable) -> None:
		"""Method that ask directory to the user and updates the `text_variable` from an Entry.
		
		Keyword arguments
		-----------------
		text_variable : tk.StringVar
			Tkinter string variable that will be updated with the user input.
		"""
		dir = askdirectory(parent=self)
		if dir:
			text_variable.set(dir)

	def select_gt_form(self) -> None:
		"""Method that ask a ground-truth filename to the user and updates the `ground_truth` from an Entry."""
		filetypes=(
			("MATLAB", "*.mat"),
		)
		file = askopenfilename(parent=self, filetypes=filetypes)
		if file:
			self.ground_truth.set(file)

	def select_no_gt_form(self):
		self.ground_truth.set('')

	def notebook_tab_changed(self) -> None:
		"""Method that adds an event to show/hide the reload button."""
		notebook = self.children['!notebook']
		if 'resume' in notebook.select():
			# packs the button widget
			self.children['btn_reload_options'].pack(expand=False, fill='x', side='left')
		else:
			# forgets the pack of the button widget
			self.children['btn_reload_options'].pack_forget()

	def turn_on_all_buttons(self) -> None:
		"""Method that implements the behaviour of the `all` button."""
		if self.all.get():
			self.run_dataset.set(True)
			self.run_gt.set(True)
			self.run_pred.set(True)
			self.run_render.set(True)
			self.run_plot_render.set(True)
			self.none.set(False)

	def turn_off_all_buttons(self) -> None:
		"""Method that implements the behaviour of the `none` button."""
		if self.none.get():
			self.run_dataset.set(False)
			self.run_gt.set(False)
			self.run_pred.set(False)
			self.run_render.set(False)
			self.run_plot_render.set(False)
			self.all.set(False)

	def change_form_state(self) -> None:
		"""Method that implements the normal buttons behaviour."""
		self.all.set(False)
		self.none.set(False)

class Demo:
	"""Implementation of the demo and all its use cases."""
	def __init__(self) -> None:
		self.resume_kwargs_filename = 'demo_kwargs.json'

		# creates a dict with the keys and its default values
		self.kwargs = {
			"input_type": '',
			"video_path": '',
			"bottom_face_path": '',
			"right_eye_path": '',
			"left_eye_path": '',
			"dataset_path": '',
			"gt_path": '',
			"out_path": '',
			"detector_type": 'FaceMesh',
			"use_threshold": True,
			"use_frame_stabilizer": False,
			"use_video_stabilizer": False,
			"neural_net_name": 'Meta-rPPG',
			"location": 'local',
			"location_path": '',
			"image_extension": 'png',
			"video_fps": 30,
			"ppg_fps": 30,
			"bottom_face_thr_type": 'replace',
			"right_eye_thr_type": 'mean',
			"left_eye_thr_type": 'mean',
			"use_magnification": False,
			"use_postprocessing": False,
			"resume_folder": "",
			"video_json_path": '',
			"status_code": 200,
			"run_dataset": True,
			"run_gt": True,
			"run_pred": True,
			"run_render": True,
			"run_plot_render": True
		}

		# Parse all the arguments that the user chose
		opt = self.parser()

		resume_kwargs = None
		# if the user wants to reload some options:
		if 'resume_folder' in opt:
			resume_kwargs_path = os.path.join(opt.resume_folder, self.resume_kwargs_filename)
			if os.path.isfile(resume_kwargs_path):
				with open(resume_kwargs_path, 'r') as fp:
					# load the options with the path passed by the user 
					resume_kwargs = json.load(fp)

		# for each key:
		for arg in self.kwargs.keys():
			# if the user wants to reload some options:
			if resume_kwargs:
				# update the defaults
				self.kwargs[arg] = resume_kwargs[arg]

			# if the user wants to update some args:
			if arg in opt:
				# gets the `arg` attribute from the `opt` parser
				self.kwargs[arg] = getattr(opt, arg)

	def parser(self):
		"""Creates the ArgumentParser to get the user options."""
		parser = argparse.ArgumentParser(description="Challenge pipeline demonstration")
		parser.add_argument("--input_type", type=str, help="Use entire_face, ROIs or dataset? (default: entire_face)", default=argparse.SUPPRESS)
		parser.add_argument("--video_path", type=str, help="Path to video (default: None)", default=argparse.SUPPRESS)
		parser.add_argument("--bottom_face_path", type=str, help="Path to bottom face video (default: None)", default=argparse.SUPPRESS)
		parser.add_argument("--right_eye_path", type=str, help="Path to right eye video (default: None)", default=argparse.SUPPRESS)
		parser.add_argument("--left_eye_path", type=str, help="Path to left eye video (default: None)", default=argparse.SUPPRESS)
		parser.add_argument("--dataset_path", type=str, help="Path to a dataset folder (default: None)", default=argparse.SUPPRESS)
		parser.add_argument("--gt_path", type=str, help="Path to the video's heart rate ground truth (default: None)", default=argparse.SUPPRESS)
		parser.add_argument("--out_path", type=str, help="Output folder path (default: None)", default=argparse.SUPPRESS)
		parser.add_argument("--detector_type", type=str, help="Facial detector (default: 'FaceMesh')", default=argparse.SUPPRESS)
		parser.add_argument("--use_threshold", action="store_true", help="To use threshold stabilization method? (default: True)", default=argparse.SUPPRESS)
		parser.add_argument("--use_frame_stabilizer", action="store_true", help="To use stabilization frame by frame? (default: False)", default=argparse.SUPPRESS)
		parser.add_argument("--use_video_stabilizer", action="store_true", help="To use stabilization on the entire video? (default: False)", default=argparse.SUPPRESS)
		parser.add_argument("--neural_net_name", type=str.lower, help="The name of the neural network that has to be used (default: 'Meta-rPPG')", default=argparse.SUPPRESS)
		parser.add_argument("--location", type=str, help="The location that the neural network has to be runned (default: 'local')", default=argparse.SUPPRESS)
		parser.add_argument("--location_path", type=str, help="The IP/PATH from the server/location that will run/load the neural network with the video (default: None)", default=argparse.SUPPRESS)
		parser.add_argument("--image_extension", type=str.lower, help="Image format that should be used when saving artificial images on disk (default: 'PNG')", default=argparse.SUPPRESS)
		parser.add_argument("--video_fps", type=int, help="If the input is a sequence of images, the user must inform the FPS of the video (default: 30)", default=argparse.SUPPRESS) 
		parser.add_argument("--ppg_fps", type=int, help="If there is a ground truth, then the user must inform its FPS (default: 30)", default=argparse.SUPPRESS)
		parser.add_argument("--bottom_face_thr_type", type=str.lower, help="The threshold type is the behaviour that happens when it is exceeded. Could be 'mean' or 'replace' (default: 'replace')", default=argparse.SUPPRESS)
		parser.add_argument("--right_eye_thr_type", type=str.lower, help="The threshold type is the behaviour that happens when it is exceeded. Could be 'mean' or 'replace' (default: 'mean')", default=argparse.SUPPRESS)
		parser.add_argument("--left_eye_thr_type", type=str.lower, help="The threshold type is the behaviour that happens when it is exceeded. Could be 'mean' or 'replace' (default: 'mean')", default=argparse.SUPPRESS)
		parser.add_argument("--use_magnification", action="store_true", help="To use video magnification? (default: False)", default=argparse.SUPPRESS)
		parser.add_argument("--use_postprocessing", action="store_true", help="Boolean responsible to define if the postprocessing must be applied to the PPG or not (default: False)", default=argparse.SUPPRESS)
		parser.add_argument("--resume_folder", type=str, help="Resume from previous output folder path (default: None)", default=argparse.SUPPRESS)
		parser.add_argument("--video_json_path", type=str, help="Resume from previous output JSON (default: None)", default=argparse.SUPPRESS)
		parser.add_argument("--run_dataset", action="store_true", help="Boolean responsible to define if the dataset creator must run again or not (default: True)", default=argparse.SUPPRESS)
		parser.add_argument("--run_gt", action="store_true", help="Boolean responsible to define if the ground-truth processing must run again or not (default: True)", default=argparse.SUPPRESS)
		parser.add_argument("--run_pred", action="store_true", help="Boolean responsible to define if the prediction must run again or not (default: True)", default=argparse.SUPPRESS)
		parser.add_argument("--run_render", action="store_true", help="Boolean responsible to define if the video rendering must run again or not (default: True)", default=argparse.SUPPRESS)
		parser.add_argument("--run_plot_render", action="store_true", help="Boolean responsible to define if the plot rendering must run again or not (default: True)", default=argparse.SUPPRESS)

		return parser.parse_args()

	def create_artificial_video(self, video_path, out_path, detector_type, use_threshold, 
								use_frame_stabilizer, use_video_stabilizer, dataset, bottom_face_thr_type,
								right_eye_thr_type, left_eye_thr_type, use_magnification, *args, **kwargs):
		"""Method that runs the `process_video` from the artificial dataset creator.
		
		Keyword arguments
		-----------------
		Check the `process_video` keyword arguments.

		Return
		------
		video_json : dict
			Dictionary that contains ROI names as keys and a list of paths as values. E.g.:
			>>> {
			>>> 	'bottom_face': ['path1/to/frame0.png', 'path1/to/frame1.png', ...],
			>>> 	'right_eye': ['path2/to/frame0.png', 'path2/to/frame1.png', ...],
			>>> 	'left_eye': ['path3/to/frame0.png', 'path3/to/frame1.png', ...],
			>>> 	'middle_face': ['path4/to/frame0.png', 'path4/to/frame1.png', ...],
			>>> }
		"""
		kwargs = {
			"video_path":           video_path, 
			"out_path":             out_path, 
			"detector_type":        detector_type,
			"save_log":             True,
			"save_videos":          True,
			"realtime":             False,
			"use_threshold":        use_threshold,
			"distortion_fixed":     False,
			"use_frame_stabilizer": use_frame_stabilizer,
			"use_eyes_lock":        False,
			"use_video_stabilizer": use_video_stabilizer,
			"threshold_modes":      {
				'bottom_face':      bottom_face_thr_type,
				'right_eye':        right_eye_thr_type, 
				'left_eye':         left_eye_thr_type, 
				# 'middle_face':      'mean'
			},
			"dataset":              dataset,
			"use_magnification":    use_magnification,
		}

		return process_video(**kwargs)

	def decompose_roi_videos(self, bottom_face_path, right_eye_path, left_eye_path, 
							out_path, dataset, video_fps, *args, **kwargs):
		"""Method that changes the video FPS and saves its frames at the disk as images.
		
		Keyword arguments
		-----------------
		bottom_face_path : str
			Path to bottom face video.

		right_eye_path : str
			Path to right eye video.

		left_eye_path : str
			Path to left eye video.

		out_path : str
			Output folder path

		dataset : Dataset
			The `dataset` object must belong to a class that enherit the `Dataset` class. 
			The `dataset` object class must have implemented the `save_video` method.
	
		video_fps : int
			FPS of the video

		Return
		------
		video_json : dict
			Dictionary that contains ROI names as keys and a list of paths as values. E.g.:
			>>> {
			>>> 	'bottom_face': ['path1/to/frame0.png', 'path1/to/frame1.png', ...],
			>>> 	'right_eye': ['path2/to/frame0.png', 'path2/to/frame1.png', ...],
			>>> 	'left_eye': ['path3/to/frame0.png', 'path3/to/frame1.png', ...],
			>>> }
		"""
		rois = ['bottom_face', 'right_eye', 'left_eye']
		paths = [bottom_face_path, right_eye_path, left_eye_path]
		dataset_json = {}

		# for each ROI
		for roi, video_path in zip(rois, paths):
			dataset_json[roi] = []

			# create the out dirs if they doesn't exist
			roi_out_path = os.path.join(out_path, roi)
			if not os.path.exists(out_path):
				os.mkdir(out_path)
			if not os.path.exists(roi_out_path):
				os.mkdir(roi_out_path)

			# convert the video to the FPS specified by the user
			if os.path.isfile(video_path):
				with VideoFileClip(video_path) as clip:
					video_path = os.path.join(out_path, f'{roi}_{video_fps}fps.mp4')
					# write it at the disk
					clip.write_videofile(video_path,fps=video_fps)

			currentframe = 0
			cam = cv2.VideoCapture(video_path)

			while True:
				ret, frame = cam.read()

				# if `cv2.VideoCapture` is no longer capturing anything: exit the infinite loop
				if ret == False:
					break

				# save the frame at the disk and append its path to the ROI list
				dataset_json[roi].append(dataset.save_video(frame, roi_out_path, currentframe))
				currentframe += 1

			cam.release()

		return dataset_json

	def write_video_json(self, video_json: dict, gt_path, video_json_path, run_gt, out_path, *args, **kwargs) -> None:
		"""Method that saves the `dataset_json` at a JSON file.
		
		Keyword arguments
		-----------------
		video_json : dict
			Dictionary that contains ROI names as keys and a list of paths as values. E.g.:
			>>> {
			>>> 	'bottom_face': ['path1/to/frame0.png', 'path1/to/frame1.png', ...],
			>>> 	'right_eye': ['path2/to/frame0.png', 'path2/to/frame1.png', ...],
			>>> 	'left_eye': ['path3/to/frame0.png', 'path3/to/frame1.png', ...],
			>>> 	'middle_face': ['path4/to/frame0.png', 'path4/to/frame1.png', ...],
			>>> }

		gt_path : str
			Path to the video's heart rate ground truth.

		video_json_path : str
			Output path that the JSON should be writed.

		run_gt : bool
			Boolean responsible to define if the ground-truth processing must run again or not.

		out_path : str
			Output folder path were the preprocessed ground-truth should be.
		"""
		if run_gt and os.path.isfile(gt_path):
			video_json['PPG'] = os.path.join(out_path, 'pulseOx.mat')
		else:
			video_json['PPG'] = gt_path

		dataset_json = {'pipeline_video': video_json}
		# Write Dataset JSON on the disk
		with open(video_json_path, 'w') as dataset_json_file:
			json.dump(dataset_json, fp=dataset_json_file, indent=1)

	def run_neural_network(self, video_json_path, out_path, neural_net_name, gt_path, 
							location_path, ppg_fps, phase, video_fps, *args, **kwargs) -> None:
		"""Method that runs a network according to existence or not from a ground-truth and with the choice of user.

		Keyword arguments
		-----------------
		video_json_path : str
			Path to the 'dataset_info.json' file that the network should read.
		
		out_path : str
			Output folder path

		neural_net_name : str
			The name of the neural network that has to be used.

		gt_path : str
			Path to the video's heart rate ground truth.
		
		location_path : str
			The IP/PATH from the server/location that will run/load the neural network with the video.

		ppg_fps : int
			FPS that the PPG was recorded.

		phase : str
			It could be 'eval' or 'test'. It will depend if there is a grount-truth or not.
		"""
		test_func = None  # this object will be a function that could be 'test' or 'eval' from 'evm' or 'meta-rppg'
		weight_path = location_path

		# If the user chose EVM
		if neural_net_name == 'evm':
			if os.path.exists(gt_path):
				# if there is a ground-truth passed by the user
				opt_instance = EvalOptions()  # get eval options
				test_func = evm_eval  
			else:
				# if there isn't a ground-truth passed by the user
				opt_instance = TestOptions()  # get test options
				test_func = evm_test
			opt = opt_instance.gather_options()
			opt.model = 'evm_cnn'
			opt.epoch = os.path.basename(weight_path).rstrip('.pth').split('_')[0] # 'latest'
			opt.gpu_ids = '-1'  # -1 for cpu ; 0 for GPU at the first position and there goes
			opt.flip = False
			opt.ppg_fps = ppg_fps
			# Elif the user chose Meta-rPPG
		elif neural_net_name == 'meta-rppg':
			opt = TrainOptions().get_known_options()
			# if there is a ground-truth passed by the user or not
			test_func = meta_eval if os.path.exists(gt_path) else meta_test
			opt.load_file = os.path.basename(weight_path).rstrip('.pth').split('_')[0] # 'latest'
			opt.fewshots = 0
			opt.continue_train = False
			opt.batch_size = 1
			opt.is_raw_dataset = True
			opt.gpu_ids = None
		else:
			print(f'The chosen network is not valid: {neural_net_name}\nIt can be "evm" or "meta-rppg"')

		opt.do_not_split_for_test = False  # we want that there is a split for test
		opt.checkpoints_dir = os.path.dirname(os.path.dirname(weight_path))
		opt.name = os.path.basename(os.path.dirname(weight_path))
		opt.feature_image_path = video_json_path
		opt.results_dir = out_path
		opt.phase = phase
		opt.video_fps=video_fps

		if neural_net_name == 'evm':
			opt = opt_instance.parse(opt)

		test_func(opt)  # run prediction

	def render_plot(self, fps, gt, pred, out_path, window_size=50, label='PPG', x_label='Time (s)') -> None:
		"""Method that render the plot and saves it at the disk.

		Keyword arguments
		-----------------
		fps : int
			FPS that the PPG was recorded.

		gt : np.ndarray or list
			Ground truth data that will be ploted.

		pred : np.ndarray or list
			Predicted data that will be ploted.
		
		out_path : str
			Output folder path that the rendered plot will be writed.

		window_size : int
			Max size of the window that should be displayed.

		label : str
			The label that will say what the ground-truth and the predicted data refer to.

		x_label : str
			The label that will indicate whats the x-axis.
		"""
		figsize = (3, 2)
		scale = 2
		fig = plt.figure(dpi=100, figsize=figsize, tight_layout=True)
		ax = plt.axes()

		# creates plots placeholders
		if gt is not None:
			gt_line, = ax.plot([], [], label=f'Ground-Truth ({label})', color='blue')
		pred_line, = ax.plot([], [], label=f'Prediction ({label})', color='orange')
		ax.set_autoscaley_on(True)

		ax.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode=None, ncol=2)
		ax.set_xlabel(x_label)
		ax.set_ylabel(label)

		def data_generator(data):
			"""Data generator that will yield the ground-truth and/or prediction data"""
			x, y = [], []
			for i, point in enumerate(data):
				x.append(i)
				y.append(point)

				# remove the first data to keep the window size
				if i > window_size:
					x.pop(0)
					y.pop(0)

				yield x, y

		pred_gen = data_generator(pred)
		if gt is not None:
			gt_gen = data_generator(gt)

		def init():
			"""Init the plot"""
			pred_line.set_data([], [])
			if gt is not None:
				gt_line.set_data([], [])
				# `relim` and `autoscale_view` to update the axis to show the data
			ax.relim()
			ax.autoscale_view()
			# Returning ground truth and predicted line
			if gt is not None:
				return pred_line, gt_line
			return pred_line,

		def update_text(name, val, position_x):
			val = int(val)
			val = str(val)

			if name == "BPM":
				color = "orange"
				minus = 0.16
			else:
				color = "blue"
				minus = 0.14
				
			# Value Box
			text = ax.text(position_x, 1.25, name + ': ' + val, 
						bbox={'facecolor': 'white', 'alpha': 1, 'pad': 4},
						verticalalignment='top', horizontalalignment='right',
						transform=ax.transAxes,
						color='black', fontsize=5)
			# Color Box
			text = ax.text(position_x - minus, 1.25, '        ', 
						bbox={'facecolor': color, 'alpha': 1, 'pad': 0},
						verticalalignment='top', horizontalalignment='right',
						transform=ax.transAxes,
						color='black', fontsize=5)
			return text

		def animate(frame_number):
			"""Get the ground-truth and/or prediction data and plot it"""
			x, y = next(pred_gen)
			pred_line.set_data(x, y)
			if gt is not None:
				x_gt, y_gt = next(gt_gen)
				gt_line.set_data(x_gt, y_gt)
			ax.relim()
			ax.autoscale_view()
			if label == 'BPM':
				if gt is not None:
					text = update_text('BPM',y[-1], 0.77)
				else:
					text = update_text('BPM',y[-1], 0.25)
			if gt is not None:
				if label == 'BPM':
					text_gt = update_text('GT',y_gt[-1], 0.25)
				return pred_line, gt_line
			return pred_line,

		# animate with the interval of 1000/fps. E.G.: 1000//30 -> 33 ms
		anim = FuncAnimation(fig, animate, init_func=init, interval=1000//fps, blit=True, repeat=False, frames=len(pred)-1)
		
		filename = label.lower()
		filename = filename.replace(' ', '_')
		filename = filename.replace('.', '_')

		anim.save(
			filename=os.path.join(out_path, f'{filename}.mp4'),
			fps=fps, 
			extra_args=['-vcodec', 'mpeg4', '-s', f'{figsize[0]*96*scale}x{figsize[1]*96*scale}'],  # ffv1
			writer='ffmpeg',
			dpi=300,
			progress_callback=lambda i, n: print(f'\rSaving frame {i+1} of {n}', end='')
		)
		print()  # New line print

	def save_kwargs(self, kwargs) -> None:
		"""Method that saves the kwargs of the Demo object at the disk"""
		with open(os.path.join(kwargs['out_path'], self.resume_kwargs_filename), 'w') as dataset_json_file:
			json.dump(kwargs, fp=dataset_json_file, indent=1)

	def run(self):
		"""Method that runs the Demo pipeline."""
		print('Args:', json.dumps(self.kwargs, indent=4))

		if self.kwargs['input_type'] != 'dataset':
			return self.run_single_video(self.kwargs)
		else:
			dataset = DemoBasicDataset(image_extension=self.kwargs['image_extension'])
			videos = dataset.map_dataset(
				base_dataset_path=self.kwargs['dataset_path'],
				subdir_name=None,
				video_name_pattern='vid.avi'
			)

			print(json.dumps(videos, indent=4))

			for video in videos:
				kwargs = self.kwargs.copy()
				kwargs['input_type'] = 'entire_face'
				kwargs['video_path'] = video['video_path']
				kwargs['gt_path'] = video['ppg_path']
				kwargs['out_path'] = os.path.join(self.kwargs['out_path'], video['subject_folder_name'])

				print('Args:', json.dumps({
					'input_type': kwargs['input_type'],
					'video_path': kwargs['video_path'],
					'gt_path': kwargs['gt_path'],
					'out_path': kwargs['out_path']
				}, indent=4))

				self.run_single_video(kwargs)
			return self.kwargs

	def run_single_video(self, kwargs):
		"""Method that runs the Demo pipeline for a single video.

		Return
		------
		kwargs : dict
			Dict with all options loaded and chosen by the user and used by the Demo.
		"""
		kwargs['phase'] = 'eval' if os.path.exists(kwargs['gt_path']) else 'test'

		try:            
			dataset = MRNirp(image_extension=kwargs['image_extension'])
			
			# if there is no resume file:
			if not os.path.isfile(kwargs['video_json_path']):
				kwargs['video_json_path'] = os.path.join(kwargs['out_path'], "dataset_info.json")

			make_dir(kwargs['out_path'])
			if kwargs['run_dataset']:
				print('Running Dataset Preprocessing')
				if kwargs['input_type'] == 'entire_face':
					video_json = self.create_artificial_video(dataset=dataset, **kwargs)
				elif kwargs['input_type'] == 'rois':
					video_json = self.decompose_roi_videos(dataset=dataset, **kwargs)
				else:
					raise Exception(f'The specified video input type has not been mapped: {kwargs["input_type"]}')

				# saves the recent created video JSON at the disk
				self.write_video_json(video_json, **kwargs)
				kwargs["run_dataset"] = False  # Performed and does not need to run again (automatically)
			else:
				if kwargs['run_render']:  # it loads a video_json
					with open(kwargs['video_json_path'], 'r') as fp:
						video_json = json.load(fp)['pipeline_video']

			# preprocess the ground-truth if it exists
			if kwargs['run_gt'] and os.path.isfile(kwargs['gt_path']):
				print('Running PPG Preprocessing')
				dataset.ppg_flatten(
					pulse_path=kwargs['gt_path'],
					out_path=kwargs['out_path'],
					ppg_fps=kwargs['ppg_fps']
				)
				kwargs['run_gt'] = False
		except Exception as e:
			kwargs['status_code'] = 501  # the 'run_dataset' or 'run_gt' has failed
			self.save_kwargs(kwargs)
			print(e.args)
			print(logging.error(logging.traceback.format_exc()))
			return kwargs

		try:
			if kwargs['run_pred']:
				print('Running Prediction')
				if kwargs['location'].lower() == 'local':
				   self.run_neural_network(**kwargs)
				   kwargs["run_pred"] = False
		except Exception as e:
			kwargs['status_code'] = 502 # the 'run_pred' has failed
			self.save_kwargs(kwargs)
			print(e.args)
			print(logging.error(logging.traceback.format_exc()))
			return kwargs

		try:
			if kwargs["run_plot_render"]:  # plot graphs
				print('Rendering graphics')
				# gets the base_folder path using the `kwargs['location_path']` to get the arrays
				nn_name = os.path.basename(os.path.dirname(kwargs['location_path']))
				model_epoch_used = os.path.basename(kwargs['location_path']).rstrip('.pth').split('_')[0]
				base_folder = os.path.join(kwargs['out_path'], nn_name, f'{kwargs["phase"]}_{model_epoch_used}', 'images')

				pred_folder = os.path.join(base_folder, 'predicted')
				ppg_pred = np.load(os.path.join(pred_folder, 'pipeline_video_ppg.npy'))

				if kwargs["use_postprocessing"]:
					ppg_pred = neurokit2.ppg_clean(ppg_pred, sampling_rate=kwargs['ppg_fps'])
					ppg_pred, _ = postprocessing(ppg_pred, current_rate=kwargs['ppg_fps'], desired_rate=30, remove_rejected=True)
					bpm_pred = ppg_to_bpm(ppg_pred)
				else:
					bpm_pred = np.load(os.path.join(pred_folder, 'pipeline_video_bpm.npy'))

				ppg_gt = None
				bpm_gt = None
				if os.path.exists(kwargs['gt_path']):
					gt_folder = os.path.join(base_folder, 'ground_truth')
					ppg_gt = np.load(os.path.join(gt_folder, 'pipeline_video_ppg.npy'))
					bpm_gt = np.load(os.path.join(gt_folder, 'pipeline_video_bpm.npy'))

				# render the PPG with fps=kwargs['ppg_fps'] and window_size=50
				self.render_plot(kwargs['ppg_fps'], ppg_gt, ppg_pred, kwargs['out_path'], window_size=50, label='PPG', x_label='Samples')
				# render the BPM with fps=1 and window_size=30
				self.render_plot(1, bpm_gt, bpm_pred, kwargs['out_path'], window_size=30, label='BPM', x_label='Time (s)')
				kwargs["run_plot_render"] = False
		except Exception as e:
			kwargs['status_code'] = 503 # some of the renders failed
			self.save_kwargs(kwargs)
			print(e.args)
			print(logging.error(logging.traceback.format_exc()))
			return kwargs

		try:
			if kwargs['run_render']:  # render the ROI cropped videos and saves it on the disk
				print('Rendering regions of interest')
				bottom_face_path = os.path.dirname(video_json['bottom_face'][0])
				create_features_video(
					os.path.join(bottom_face_path, 'Frame%05d.'+kwargs['image_extension']), 
					os.path.dirname(bottom_face_path), 
					'bottom_face.avi', 
					out_size=(400, 400), 
					out_fps=kwargs['video_fps']
				)
				right_eye_path = os.path.dirname(video_json['right_eye'][0])
				create_features_video(
					os.path.join(right_eye_path, 'Frame%05d.'+kwargs['image_extension']), 
					os.path.dirname(right_eye_path), 
					'right_eye.avi', 
					out_size=(100, 100), 
					out_fps=kwargs['video_fps']
				)
				left_eye_path = os.path.dirname(video_json['left_eye'][0])
				create_features_video(
					os.path.join(left_eye_path, 'Frame%05d.'+kwargs['image_extension']), 
					os.path.dirname(left_eye_path), 
					'left_eye.avi',     
					out_size=(100, 100), 
					out_fps=kwargs['video_fps']
				)
				kwargs["run_render"] = False
		except Exception as e:
			kwargs['status_code'] = 504 # Rendering of regions of interest failed
			self.save_kwargs(kwargs)
			print(e.args)
			print(logging.error(logging.traceback.format_exc()))
			return kwargs

		kwargs['status_code'] = 200  # If you have executed here without exceptions: status code 200
		self.save_kwargs(kwargs)
		return kwargs

if __name__ == '__main__':
	demo = Demo()
	# demo.opt.video_path = '/home/desafio01/Documents/Codes/bio_hmd/Dataset_MR_NIRP/MR-NIRP_Indoor/Subject1_still_940/RGB_20s/Frame%05d.pgm'
	# demo.opt.out_path = '/home/desafio01/Documents/Codes/bio_hmd/Dataset_MR_NIRP/dataset_out_pipeline'
	# demo.opt.gt_path = '/home/desafio01/Documents/Codes/bio_hmd/Dataset_MR_NIRP/artificial_v2/Subject1_still_940/RGB_demosaiced/pulseOx_bpm.mat'
	demo.run()
