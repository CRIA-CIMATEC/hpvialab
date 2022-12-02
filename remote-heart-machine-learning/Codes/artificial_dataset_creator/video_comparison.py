#! /usr/bin/python
# -*- coding: utf-8 -*-
# Code adapted from https://gist.github.com/aahmd/921ff61249e1377bb6617cadc88e8f21
"""vlc media player; based off example in vlc repo:
`http://git.videolan.org/?p=vlc/bindings/python.git;a=commit;h=HEAD`
See also:
`http://infohost.nmt.edu/tcc/help/pubs/tkinter/web/menu.html`
`http://infohost.nmt.edu/tcc/help/pubs/tkinter/web/menu-coptions.html`
"""
from tkinter.constants import INSERT, NSEW
from utils import FrameSequence

from biohmd import DemoForm
import json
import math
import vlc
import time
import os
import tkinter as tk
from tkinter import ttk, Label
from tkinter.filedialog import askopenfilename, asksaveasfile
from PIL import Image, ImageTk
import sys

class Container:
	"""Class container containing a tkinter frame and the vlc video player"""
	
	def __init__(self, frame: tk.Frame, video_path: os.PathLike=None, vlc_media_instance=None, \
				video_label=None, is_graph=False):
		"""
		Keyword arguments
		-----------------
		frame : tk.Frame
			Tkinter frame containing a video

		video_path : str or os.PathLike
			Video path that can be used as label.

		vlc_media_instance : object returned by `media_player_new()`
			Instance of VLC Media Player that is used to play the video.

		video_label : str
			Label that appears above the video, usually comes from JSON.

		is_graph : bool
			Boolean responsible to identify the containers that present a graph as a video.
		"""
		
		self.frame = frame
		self.vlc_media_instance = vlc_media_instance
		self.video_path = video_path
		self.graph_name = None
		self.is_graph = is_graph

		# if is graph:
		if isinstance(video_label, str) and is_graph:
			if video_label.isalnum():
				# fill the graph_name
				self.graph_name = video_label.lower()

		if is_graph:
			assert self.graph_name in ['ppg', 'bpm'], "Graph name is not ppg or bpm"

		text = tk.Entry(frame)

		# If video_label is not None:
		if video_label:
			# Adds it on label above the video
			text.insert(INSERT, video_label)
		else:
			if video_path:
				# Adds the end of the video path on label above the video
				text.insert(INSERT, os.sep.join(video_path.split(os.sep)[-2:]))
			else:
				text.insert(INSERT, ' ')

		# Fill only the X-axis and fit up the video
		text.pack(side='top', fill='x', expand=False)

	def update_scale(self):
		"""Method that updates the video scale inside the frame

		- Verifies if the width and height of the video are larger than the width and the height of the frame
		- Updates the scale so that the video is not cut by the frame
		"""
		if self.vlc_media_instance:
			if (self.frame.winfo_width() < self.vlc_media_instance.video_get_width(0)) or \
				(self.frame.winfo_height() < self.vlc_media_instance.video_get_height(0)):
				self.vlc_media_instance.video_set_scale(0.) # Keep the video on the original scale
			else:
				self.vlc_media_instance.video_set_scale(1.) # Allows the resizing of the video to fit in the window

class PyPlayer(tk.Frame):
	"""PyPlayer class containing the menu, the VLC frame set and the progress bar"""
	def __init__(self, container, container_instance: tk.Tk) -> None:
		"""
		Keyword arguments
		-----------------
		container : BaseTkContainer
			Base class responsible to implement basic configurations

		container_instance : tk.Tk
			TK instance that belongs to root
		"""
		tk.Frame.__init__(self, container_instance)
		self.container = container
		self.container_instance = container_instance

		# main Frame that will have the videos
		self.container_playlist = tk.Frame(container_instance, background='black')
		self.container_playlist.pack(side='top', expand=True, fill='both')

		# two Frames representing the rows of the grid
		self.row_frames = [tk.Frame(self.container_playlist), tk.Frame(self.container_playlist)]

		self.containers = []

		# main instance of the VLC that will create an instance for each video
		self.vlc_instance = self.create_vlc_instance()
		
		# main menubar
		self.menubar = tk.Menu(self.container_instance)

		self.menubar.add_command(label="Add Video", command=self.add_video, accelerator="ctrl + o")
		self.menubar.add_command(label="Process Video", command=self.launch_demo_form)
		self.menubar.add_command(label="Load JSON", command=self.load_json)
		self.menubar.add_command(label="Save JSON", command=self.save_json)

		# controls
		self.menubar.add_command(label="Pause", command=self.pause)
		self.menubar.add_command(label="Play", command=self.play)
		self.menubar.add_command(label="Stop", command=self.stop)

		# BPM/PPG buttons to show/hide the graphs
		menu = tk.Menu(self.menubar, relief=tk.RAISED)
		self.show_ppg = tk.IntVar(value=1)
		self.show_bpm = tk.IntVar(value=1)
		menu.add_checkbutton(label='PPG', variable=self.show_ppg, command=self.change_graph_video_status)
		menu.add_checkbutton(label='BPM', variable=self.show_bpm, command=self.change_graph_video_status)
		self.menubar.add_cascade(label='Plot options', menu=menu)

		# other menus
		self.menubar.add_command(label="Clear Videos", command=self.clear_videos)
		self.menubar.add_command(label="Debug", command=self._debug)
		self.menubar.add_command(label="Quit", command=self.close)

		self.container_instance.config(menu=self.menubar)

		# panel that will have the video controls
		self.control_panel = tk.Frame(container_instance, background='black')
		self.control_panel.pack(side='bottom')

		# Play/Pause buttom bottom left
		self.buttom_switch()

		# Slider and elapsed time label
		self.slide_label()

		self.demo = None
		self.event_end_triggered = False

	def slide_label(self):
		'''METHOD FOR PROGRESS BAR'''
		# Slider Customization and Configuration
		current_value = tk.DoubleVar()

		def get_current_value():
			converted_current_time = time.strftime('%M:%S', time.gmtime(self.slider.get()))
			return converted_current_time

		def slider_changed(event):
			elapsed_time_label.configure(text=get_current_value())

		self.slider = tk.Scale(self.control_panel, 
								from_ = 0, to = 100, 
								orient = 'horizontal', 
								length = self.control_panel.winfo_screenwidth(),
								variable = current_value,
								command = slider_changed,
								showvalue=False,
								borderwidth=0,
								relief=tk.FLAT,
								sliderrelief=tk.FLAT,
								troughcolor='#000',
								)

		# Label with the value of the elapsed time
		elapsed_time_label = Label(self.control_panel, text = '--:--')
		elapsed_time_label.pack(side='right')
		
		self.slider.pack(side='right')

	def buttom_switch(self):
		# Play/Pause Button
		global is_on
		is_on = True

		# SETTING BUTTONS IMAGES
		icon_play_button = Image.open("buttons/PauseButton.png") 
		icon_pause_button = Image.open("buttons/PlayButton.png")
		# RESIZE BUTTONS IMAGES
		resized_icon_play_button = icon_play_button.resize((10, 10))
		resized_icon_pause_button = icon_pause_button.resize((10, 10))
		button_play = ImageTk.PhotoImage(resized_icon_play_button)
		button_pause = ImageTk.PhotoImage(resized_icon_pause_button)

		def switch():
			'''Method to use buttons Play/Pause'''
			global is_on
			# Conditional for use play and pause
			if is_on == True:
				img_label.config(image=button_pause)
				self.pause()
				is_on = False
			else:
				img_label.config(image=button_play)
				self.play()
				is_on = True

		img_label = tk.Button(self.control_panel, image=button_play, command = switch)
		img_label.image = button_play
		img_label.pack(side='left')

	def launch_demo_form(self) -> None:
		"""Method that is called to open the form to the user"""
		self.demo = DemoForm(self)
		# keep the focus of the window of this form to avoid many instances at the same time
		self.wait_window(self.demo)

	def show_results(self, out_path, status_code, *args, **kwargs) -> None:
		"""Method that opens all videos after all processings of the Demo.
		
		Keyword arguments
		-----------------
		out_path : str
			Output path that contains the results.

		status_code : int
			Integer representing the status of the Demo's processing:
				200: The execution ended successfully.
				501: The execution ended with an error in the preprocessing of the video or the ground-truth.
				502: The execution ended with an error in the network prediction.
				503: The execution ended with an error in the rendering of the graphs plot.
				504: The execution ended with an error in the rendering of ROI videos.
		"""
		if status_code == 200:
			self.clear_videos()  # clear the videos that are playing

			# add each ROI video to the player
			self.add_video(os.path.join(out_path, 'bottom_face.avi'))
			self.add_video(os.path.join(out_path, 'right_eye.avi'))
			self.add_video(os.path.join(out_path, 'left_eye.avi'))

			# add each graph video to the player
			self.add_video(os.path.join(out_path, 'bpm.mp4'), 'BPM', is_graph=True)
			self.add_video(os.path.join(out_path, 'ppg.mp4'), 'PPG', is_graph=True)
			
			# organize the current videos with 3 videosat the first row and 2 at the second row
			self.organize([3, 2])

	def add_video(self, video_path=None, video_label=None, is_graph=False) -> None:
		"""Method that adds a video in the cluster of the screen.
		
		Keyword arguments
		-----------------
		video_path : str
			Video path to be added.

		video_label : str
			Label that appears above the video.
		"""
		# creates the `vlc_media_instance` object that serves to control a video
		vlc_media_instance = self.vlc_instance.media_player_new()

		# creates the vlc video frame
		video_panel = ttk.Frame()
		canvas = tk.Canvas(video_panel, background='black')
		canvas.pack(fill=tk.BOTH, expand=True, side='bottom')

		if video_path:
			# if the `video_path` corresponds to an entire video
			if os.path.isfile(video_path):
				open_return = self.open(vlc_media_instance, canvas, video_path)
			# elif the `video_path` corresponds to a sequence of images
			elif '%' in video_path:
				vlc_media_instance = FrameSequence(video_path, self.demo.video_fps_entry.get(), canvas, cyclic=False)
				open_return = video_path
				# add two listeners to when the video ends and for when the video advances in time
				vlc_media_instance.event_attach(vlc.EventType.MediaPlayerEndReached, self.onEnd)
				vlc_media_instance.event_attach(vlc.EventType.MediaPlayerTimeChanged, self.media_time_changed)
		else:
			open_return = self.open(vlc_media_instance, canvas, video_path)

		# If User cancel the operation
		if not open_return:
			video_panel.grid_forget()
			video_panel.destroy()
		else:
			# assert os.path.exists(open_return), 'Caminho do vídeo não existe'
			# Add to list of videos to access later
			self.containers.append(
				Container(
					frame=video_panel, 
					vlc_media_instance=vlc_media_instance, 
					video_path=open_return, 
					video_label=video_label,
					is_graph=is_graph
					))
			# Stop videos that may be played to synchronize all
			self.stop()
			# Reorganizes the grid of the screen taking into account the new video
			self.organize()

	def organize(self, columns_dist=None) -> None:
		"""Organize the videos present in the self.container lists
		
		Keyword arguments
		-----------------
		columns_dist : list
			Distribution of columns in the two lines. E.G.: 
				[3, 2]: 3 columns at the first column and 2 columns at the second one
		"""
		# `row_size` needs to distribute the `self.containers` over the two rows
		row_size = len(self.containers) / 2

		self.clear_grid(self.row_frames)

		# it will have just one row if it has only one video
		sides = [0]
		if len(self.containers) > 1:
			sides.append(1)

		# creates a iterable over the `self.containers`
		containers = iter(self.containers)
		# `containers_heights` is used when you want to have a row bigger than the other
		containers_heights = iter([1, 1])

		if columns_dist:
			assert sum(columns_dist) == len(self.containers), "The sum of `columns_dist` must have the same size of containers"
			columns_dist = iter(columns_dist)
			# containers_heights = iter([4, 6])

		# for each row
		for line, side in zip(self.row_frames, sides):
			line.grid(row=side, column=0, sticky=NSEW)
			line.grid_rowconfigure(0, weight=1)

			# distributing the columns over the rows
			if columns_dist:
				# use the distribution given by the user
				columns = next(columns_dist)
			elif side == 0:
				# if it is the first row: use the maximum number of columns. E.G.: 3 of 5
				columns = math.ceil(row_size)
			elif side == 1:
				# if it is the second row: use the minimum number of columns. E.G.: 2 of 5
				columns = math.floor(row_size)

			for i in range(columns):
				# makes the weight of the line and the column equal to 1
				line.grid_columnconfigure(i, weight=1)

				container = next(containers, None)

				# Adds the video on line J and column I;lock it in all directions
				container.frame.grid(row=0, column=i, sticky=NSEW, in_=line)

			self.container_playlist.grid_rowconfigure(side, weight=next(containers_heights))

		self.container_playlist.grid_columnconfigure(0, weight=1)

	def open(self, vlc_media_instance, video_panel, video_path=None):
		"""New window allowing user to select a file
		
		Keyword arguments
		-----------------
		vlc_media_instance : object returned by the `media_player_new()` method
			Object that serves to control a video

		video_panel : tk.Canvas
			Tkinter widget that the video will be displayed.

		video_path : str
			Video path that the user chosen.
		
		Return
		------
		False : bool
			If the file chosen is a tuple.

		opened_video_path : str
			Video path that the user chosen.
		"""
		if video_path is None:
			file = askopenfilename(
				filetypes=(
					("Audio Video Interleave", "*.avi"),
					("Matroska", "*.mkv"),
					("MP4", "*.mp4"),
				)
			)
		else:
			file = video_path

		# file shouldn't be a tuple
		if isinstance(file, tuple):
			return False

		if os.path.isfile(file):
			directory_name = os.path.dirname(file)
			file_name = os.path.basename(file)
			self.Media = self.vlc_instance.media_new(
				str(os.path.join(directory_name, file_name))
			)
			vlc_media_instance.set_media(self.Media)
			vlc_media_instance.set_xwindow(video_panel.winfo_id())
			h = video_panel.winfo_id()
			_isWindows = hasattr(sys, 'getwindowsversion')
			if _isWindows:
				vlc_media_instance.set_hwnd(h)  # used to keep the window inside the main program

			# add two listeners to when the video ends and for when the video advances in time
			em = vlc_media_instance.event_manager()
			em.event_attach(vlc.EventType.MediaPlayerEndReached, self.onEnd)
			em.event_attach(vlc.EventType.MediaPlayerTimeChanged, self.media_time_changed)
			
			# return the opened video path
			return str(os.path.join(directory_name, file_name))

	def create_vlc_instance(self) -> vlc.Instance:
		"""Create a vlc instance; `https://www.olivieraubert.net/vlc/python-ctypes/doc/vlc.MediaPlayer-class.html`"""
		vlc_instance = vlc.Instance("--no-xlib")
		self.container_instance.update()
		return vlc_instance

	def play(self) -> None:
		"""Play all videos."""
		for video_player in self.containers:
			if video_player.vlc_media_instance:
				if video_player.vlc_media_instance.play() == -1:
					print("Couldn't play the video")
		
		for video_player in self.containers:
			# Waiting half second to update the video scale
			self.after(500, video_player.update_scale)

	def close(self) -> None:
		"""Close the window."""
		self.container.delete_window()

	def pause(self) -> None:
		"""Pause all players."""
		for video_player in self.containers:
			if video_player.vlc_media_instance:
				video_player.vlc_media_instance.pause()

	def stop(self) -> None:
		"""Stop all players."""
		for video_player in self.containers:
			if video_player.vlc_media_instance is not None:
				video_player.vlc_media_instance.stop()
		# Make the progress slider be restarted
		self.slider.set(0)

	def clear_videos(self) -> None:
		"""Removes all videos from memory."""
		# print('clear_videos event triggered')
		# Stop videos before removing frames
		self.stop()
		for video_player in self.containers:
			# Removes Added Listeners
			if video_player.vlc_media_instance:
				em = video_player.vlc_media_instance.event_manager()
				em.event_detach(vlc.EventType.MediaPlayerEndReached)
				em.event_detach(vlc.EventType.MediaPlayerTimeChanged)
			video_player.frame.destroy()

		# clear the containers list
		self.containers.clear()

		self.clear_grid(self.row_frames)
		self.container_playlist.grid_rowconfigure(0, weight=0)
		self.container_playlist.grid_rowconfigure(1, weight=0)

	def load_json(self) -> None:
		"""Method that opens new videos based on a JSON.
		JSON should contain path of the video and can contain his label.
		The order of videos disposed in JSON will be maintained when opening them."""

		file = askopenfilename(
			filetypes=[
				("Json files","*.json"),
				("All Files","*.*")]
		)
		if os.path.isfile(file):
			with open(file, 'r') as fp:
				json_data = json.load(fp)
			
			# for each video
			for attributes in json_data.values():
				# Take the path and the label of the video
				path = attributes.get('video_path', None)
				label = attributes.get('video_label', None)

				# the video will only be created if there is path
				if path:
					self.add_video(path, label)
			
	def save_json(self) -> None:
		"""Saves the list of videos in a JSON file.
		Saves the current path and the video label."""
		data_json = {}

		# for each video
		for i, video_player in enumerate(self.containers):
			video_label = None
			# Take the current label of the video (changed by the user or not)
			if type(video_player.frame.pack_slaves()[1]) == tk.Entry:
				video_label = video_player.frame.pack_slaves()[1].get()

			# adds to the json.
			data_json[f'Video_{i}'] = {
				'video_path': video_player.video_path,
				'video_label': video_label
			}

		fp = asksaveasfile(
				initialfile = 'Untitled.json', 
				defaultextension=".json", 
				filetypes=[
					("Json files","*.json"),
					("All Files","*.*")
				]
		)

		if fp:
			json.dump(data_json, fp, indent=1)
			fp.close()

	def reset_event_status(self) -> None:
		"""Method that will limit the event end to be triggered"""
		self.event_end_triggered = False

	def onEnd(self, event):
		"""Method called when videos end up playing"""
		if event.type == vlc.EventType.MediaPlayerEndReached:
			# if the event should be triggered
			if not self.event_end_triggered:
				self.event_end_triggered = True
				self.after(100, self.stop)
				self.after(500, self.play)
				self.after(5000, self.reset_event_status)

	def media_time_changed(self, event):
		"""Method called when videos advance in time"""
		if self.containers[0] and event.type == vlc.EventType.MediaPlayerTimeChanged:
			# verifies if it is greater than zero to not generate exception
			if int(self.containers[0].vlc_media_instance.get_length()/1000) > 0:
				# print('media_time_changed event triggered')
				# Updates the progress slider according to the time reproduced 
				# and total of the first video
				current_video_time = self.containers[0].vlc_media_instance.get_time()/1000
				current_video_length = self.containers[0].vlc_media_instance.get_length()/1000
				video_percentage = round(self.slider.get()/current_video_length,2)
				# Make slider have the same length as the video
				if current_video_length != self.slider['to']:
					self.slider.config(to = current_video_length)
				# Make the video navigable through the slider
				if (abs(current_video_time - self.slider.get()) > 1):
					for video_player in self.containers:
						video_player.vlc_media_instance.set_position(video_percentage)
				else:
					self.slider.set(current_video_time)

	def clear_grid(self, frame) -> None:
		"""Method that forget all video frames and reconfigures the rows"""
		for row_frame in frame:
			for coluna in range(row_frame.grid_size()[0]):
				row_frame.grid_columnconfigure(coluna, weight=0)
			for linha in range(row_frame.grid_size()[1]):
				row_frame.grid_rowconfigure(linha, weight=0)
			row_frame.grid_forget()

	def _debug(self) -> None:
		"""Debugging."""
		import pdb; pdb.set_trace()
		pass

	def change_graph_video_status(self) -> None:
		"""Method that will control the show/hid of the graphs"""
		ppg_status = bool(self.show_ppg.get())
		bpm_status = bool(self.show_bpm.get())
		ppg_container = None
		bpm_container = None

		# for each video
		for container in self.containers:
			if container.is_graph:
				if container.graph_name == 'ppg':
					ppg_container = container
				elif container.graph_name == 'bpm':
					bpm_container = container
		
		# if the user choses to turn off all graphs:
		# one of then must be on
		if not ppg_status and not bpm_status:
			# if the bpm frame is mapped:
			if bpm_container.frame.grid_info() != {}:
				bpm_status = True
				self.show_bpm.set(1)
			# if the ppg frame is mapped:
			if ppg_container.frame.grid_info() != {}:
				ppg_status = True
				self.show_ppg.set(1)

		# if the frame isn't mapped and it must be mapped
		if ppg_container.frame.grid_info() == {} and ppg_status:
			ppg_container.frame.grid()
			self.row_frames[1].grid_columnconfigure(1, weight=1)
		# if the frame is mapped and it must be unmapped
		elif ppg_container.frame.grid_info() != {} and not ppg_status:
			ppg_container.frame.grid_remove()
			if bpm_status:  # if BPM is on:
				self.row_frames[1].grid_columnconfigure(1, weight=0)
				self.row_frames[1].grid_columnconfigure(0, weight=1)

		# if the frame isn't mapped and it must be mapped
		if bpm_container.frame.grid_info() == {} and bpm_status:
			bpm_container.frame.grid()
			self.row_frames[1].grid_columnconfigure(0, weight=1)
		# if the frame is mapped and it must be unmapped
		elif bpm_container.frame.grid_info() != {} and not bpm_status:
			bpm_container.frame.grid_remove()
			self.row_frames[1].grid_columnconfigure(0, weight=1)
			if ppg_status:  # if PPG is on:
				self.row_frames[1].grid_columnconfigure(0, weight=0)
				self.row_frames[1].grid_columnconfigure(1, weight=1)

class BaseTkContainer:
	"""Base class responsible to implement basic configurations"""
	def __init__(self, title):
		"""		
		Keyword arguments
		-----------------
		title : str
			Window title
		"""
		self.tk_instance = tk.Tk()
		self.tk_instance.title(title)
		self.tk_instance.protocol("WM_DELETE_WINDOW", self.delete_window)
		
		# Sets the window size
		screenwidth = self.tk_instance.winfo_screenwidth()
		screenheight = self.tk_instance.winfo_screenheight()
		alignstr = '%dx%d' % (screenwidth, screenheight)
		self.tk_instance.geometry(alignstr) # "1920x1080" default to 1080p
		self.tk_instance.configure(background='black')
		self.theme = ttk.Style()
		self.theme.theme_use("clam")

	def delete_window(self):
		"""Close all program windows"""
		tk_instance = self.tk_instance
		tk_instance.quit()
		tk_instance.destroy()
		os._exit(1)
	
	def __repr__(self):
		return "Base tk Container"

if __name__ == '__main__':
	root = BaseTkContainer(title="Video Comparison")
	player = PyPlayer(root, root.tk_instance)
	root.tk_instance.mainloop()
