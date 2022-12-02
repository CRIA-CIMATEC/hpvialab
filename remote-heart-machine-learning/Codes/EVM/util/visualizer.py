import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
from PIL import Image
from natsort import natsorted
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr
import torch
from . import heartpy_bpm
from heartpy import hampel_filter
from heartpy.datautils import outliers_modified_z

if sys.version_info[0] == 2:
	VisdomExceptionBase = Exception
else:
	VisdomExceptionBase = ConnectionError

def window_average(x, w):
	""" Method to apply a convolution 
	Paramenters:
	x (array) -- one dimensional array 
	w (int) -- Stride value 
	"""
	return np.convolve(x, np.ones(w), 'valid')[: : w] / w

def mean_absolute_percentage_error(y, y_pred): 
	"""Calculate mean absolute percentage error
	Paramenters:
	y (array)  -- ground truth values 
	y_pred (array) -- predicted values 
	"""
	if 0 in y:
		idxs = np.where(y == 0)[0]
		y = np.delete(y, idxs)
		y_pred = np.delete(y_pred, idxs)
		
	return np.mean(np.abs((y_pred - y) / y)) * 100

def mean_error(y, y_pred):
	"""Method to return mean of ground truth values and predicted values"""
	return np.mean(y_pred - y)
	
def stand_dev(y, y_pred):
	"""Method to return standard deviatation of ground truth values and predicted values"""
	return np.std(y_pred)
  
def rmse(y, y_pred):
	"""Method to return the mean_squared_error of ground truth values and predicted values"""
	return mean_squared_error(y, y_pred, squared=False)

def pearson(y, y_pred):
	"""Method to return the Pearson coefficient between ground truth values and predicted values"""
	return pearsonr(y, y_pred)[0]
  
def mae(y, y_pred):
	"""Method to return the mean absolute error between ground truth values and predicted values"""
	return mean_absolute_error(y, y_pred)
 
def save_plot_bpm(webpage, name, y, x, stride, predicted_fs = 1):
	"""Calculate and save BPM plot 
	Parameters:
	webpage (the HTML class) -- the HTML webpage class that stores these images
	name (int)				 -- subject index
	y (array)				 -- Avarage window in y position
	x (array) 				 -- Avarage window in x position
	stride (int)			 -- Stride window
	predicted_fs(int)		 -- seconds of fps
	"""
	t = list(range(len(x)))
	fig = plt.figure(figsize=(20, 5))
	fig.suptitle(f'BPM prediction of {name}', fontsize = 20)
	plt.xlabel('Feature Image', fontsize = 18)
	
	seconds = int(stride / predicted_fs)
	
	my_xticks = [f"[{seconds * i}:{seconds * i + seconds}]" for i in range(0, len(t))]
	
	ylabel = 'BPM'
	plt.xticks(t[::stride], my_xticks[::stride])

	plt.ylabel(ylabel, fontsize = 16)
	if y is not None:
		plt.plot(t, y, label = "Ground Truth")
	plt.plot(t, x, label = "Prediction")
	plt.legend(loc="upper left")
	plt.savefig(os.path.join(webpage.get_image_dir(), f"{name}_plot.png"))

def save_plot_ppg(webpage, name, y, x, stride, predicted_fs = 1):
	"""Calculate and save PPG plot 
	Parameters:
	webpage (the HTML class) -- the HTML webpage class that stores these images
	name (int)				 -- subject index
	y (array)				 -- Avarage window in y position
	x (array) 				 -- Avarage window in x position
	stride (int)			 -- Stride window
	predicted_fs(int)		 -- seconds of fps
	"""
	t = list(range(len(x)))
	fig = plt.figure(figsize=(20, 5))
	fig.suptitle(f'PPG prediction of {name}', fontsize = 20)
	plt.xlabel('Feature Image', fontsize = 18)
	
	seconds = int(stride / predicted_fs)
	
	my_xticks = [f"[{seconds * i}:{seconds * i + seconds}]" for i in range(0, len(t))]
	
	ylabel = 'PPG'
	plt.xticks(t[::stride], my_xticks[::stride])

	plt.ylabel(ylabel, fontsize = 16)
	if y is not None:
		plt.plot(t, y, label = "Ground Truth")
	plt.plot(t, x, label = "Prediction")
	plt.legend(loc="upper left")
	plt.savefig(os.path.join(webpage.get_image_dir(), f"{name}_plot_ppg.png"))

def replace_NANs(a):
	"""Method to replace NaNs
	Parameters:
	a (array) -- receive an array"""
	ind = np.where(~np.isnan(a))[0]
	first, last = ind[0], ind[-1]
	a[:first] = a[first]
	a[last + 1:] = a[last]
	return a

def get_bpm(val, stride_window, predicted_fs, opt):
	"""
	Method to get BPM from heart rates
	Parameters:
	val (array)   -- array containing heart rate values
	stride_window (int) -- window size
	predicted_fs (int) -- seconds 
	opt (options)  -- argument that contain options
	"""
	segment_width = stride_window # (seconds)
	stride = predicted_fs # (seconds)
	fps = opt.video_fps
	bpm_out = []
	for i in range(0, val.shape[0], stride * fps):
		bpm_out.extend(heartpy_bpm.get_heartpy_bpm(val[i:i+(segment_width*fps)], fps, segment_width, stride))
	if bpm_out == []:
		return bpm_out
	elif np.all(np.isnan(bpm_out)):
		return [0]

	bpm_out = heartpy_bpm.interpolate_nan(bpm_out)
	if stride > 1:
		# interpolation in the values to maintain the quantity of seconds
		bpm_out = heartpy_bpm.interpolate_grow(bpm_out, stride=stride)
	# Interpolation of outliers values
	bpm_out = heartpy_bpm.interpolate_outliers(bpm_out, 2)

	bpm_out = replace_NANs(bpm_out)
    # Remove outliers
	bpm_out = outliers_modified_z(bpm_out)
	# Detect outliers based on hampel filter
	bpm_out = hampel_filter(bpm_out[0])

	return bpm_out

def to_cpu(x):
	"""
	Method to concatenate an array and reshape it 
	Parameters:
	x(array)  -- receive an array
	"""
	return torch.cat(x).cpu().numpy().reshape((len(x),))
	
def compute_and_save_metrics(webpage, videos, opt, gt = 'ground_truth', stride_window = 4, predicted_fs = 1):
	"""
	Method to compute and save metrics for each subject
	Parameters:
	webpage(the HTML class)     --the HTML webpage class that stores these images
	videos(dict)  				-- dict containing frames from a subject video
	opt (options)  				-- argument that contain options
	gt(str) 					-- name of the ground truth key
	stride_window (int) 		-- window size
	predicted_fs (int) 			-- seconds
	"""
	df_list = []
	for subject_idx, hr_values in videos.items():
		y = None
		if gt in hr_values.keys():
			y = hr_values.pop(gt, None)
			y = to_cpu(y)
			np.save(os.path.join(webpage.get_image_dir(), gt, f'{subject_idx}_ppg.npy'), y)
			y_bpm = get_bpm(y, stride_window, predicted_fs, opt)
		
			np.save(os.path.join(webpage.get_image_dir(), gt, f'{subject_idx}_bpm.npy'), y_bpm)
		metrics = {
			'Mean Error': mean_error,
			'Standard Deviation': stand_dev,
			'RMSE': rmse,
			'MAPE': mean_absolute_percentage_error,
			'Pearson': pearson}
		results = {"": hr_values.keys()}
		for key in hr_values:
			x = hr_values[key]
			x = to_cpu(x)
			np.save(os.path.join(webpage.get_image_dir(), key, f'{subject_idx}_ppg.npy'), x)
			x_bpm = get_bpm(x, stride_window, predicted_fs, opt)

			np.save(os.path.join(webpage.get_image_dir(), key, f'{subject_idx}_bpm.npy'), x_bpm)
			
			x = window_average(x, stride_window) if stride_window > 0 else x
			x_bpm = window_average(x_bpm, stride_window) if stride_window > 0 else x_bpm 
			if y is not None:
				y = window_average(y, stride_window) if stride_window > 0 else y
				y_bpm = window_average(y_bpm, stride_window) if stride_window > 0 else y_bpm

				results = {**results, **{key:round(metrics[key](y, x), 2) for key in metrics}}

				results = {**results, **{key:round(metrics[key](y_bpm, x_bpm), 2) for key in metrics}}

			save_plot_ppg(webpage, subject_idx, y, x, stride_window, predicted_fs)

			save_plot_bpm(webpage, subject_idx, y_bpm, x_bpm, stride_window, predicted_fs)

		df = pd.DataFrame(results)
		df_list.append(df)
		df.to_csv(os.path.join(webpage.get_image_dir(), f"{subject_idx}_metrics_eval.csv"))
	df_mean = pd.concat(df_list)
	df_mean = df_mean.mean()
	df_mean = df_mean.round(2)
	df_mean = df_mean.transpose()
	df_mean.to_csv(os.path.join(webpage.get_image_dir(), f"mean_metrics_eval.csv"))

def save_hr_values(webpage, videos, image_path, opt, denormalize = True):
	"""Method to save heart values from a video
	
	Parameters:
	webpage(the HTML class)     		--the HTML webpage class that stores these images
	videos(dict)  						-- dict containing frames from a subject video
	image_path(str)             		-- path of the images
	denormalize(bool)    				--	denormalize heart rate array
	"""
	for subject_idx, hr_values in videos.items():
		for key in hr_values.keys():
			zip_iterator = zip(image_path, hr_values[key])
			hr_dict = dict(zip_iterator)

			ordered = natsorted(hr_dict.items())

			image_dir = webpage.get_image_dir()
			short_path = ntpath.basename(image_path[0])
			name = os.path.splitext(short_path)[0]
			current_video_name = name.split("_")[0]
			
			hr_list = []
			for index, hr_tuple in enumerate(ordered, start = 1):
				label, hr_tensor = hr_tuple
				short_path = ntpath.basename(image_path[0])
				name = os.path.splitext(short_path)[0]
				temp_video_name = f'{subject_idx}_{name.split("_")[0]}'

				if current_video_name != temp_video_name or index == len(hr_dict):
					current_video_name = temp_video_name
					hr_array = np.array(hr_list, dtype = np.double)
					folder_dir = os.path.join(image_dir, key)

					save_path = os.path.join(folder_dir, current_video_name)
					np.save(save_path, hr_array)
			
			result = hr_tensor[0].cpu().float().numpy()
			result = util.denormalize_hr_array(hr_tensor[0], opt.feature_image_path, subject_idx) if denormalize else hr_tensor[0]
			hr_list.append(result)

def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
	"""Save images to the disk.

	Parameters:
		webpage (the HTML class) -- the HTML webpage class that stores these images (see html.py for more details)
		visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
		image_path (str)         -- the string is used to create image paths
		aspect_ratio (float)     -- the aspect ratio of saved images
		width (int)              -- the images will be resized to width x width

	This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
	"""
	image_dir = webpage.get_image_dir()
	short_path = ntpath.basename(image_path[0])
	name = os.path.splitext(short_path)[0]
	
	webpage.add_header(name)
	ims, txts, links = [], [], []   
	for label, im_data in visuals.items():

		im = util.tensor2im(im_data)
		
		name = name.split("_")[0] if label == "real_B" else name 
		image_name = '%s.jpg' % (name)
		
		temp = Image.open(image_path[0])
		tw, th = temp.size
		
		im = im[0:th, 0:tw] 
		
		folder_dir = os.path.join(image_dir, label)
		save_path = os.path.join(folder_dir, image_name)
		util.save_image(im, save_path, aspect_ratio=aspect_ratio)
		ims.append(image_name)
		txts.append(label)
		links.append(image_name)
	webpage.add_images(ims, txts, links, width=width)


class Visualizer():
	"""This class includes several functions that can display/save images and print/save logging information.

	It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
	"""

	def __init__(self, opt):
		"""Initialize the Visualizer class

		Parameters:
			opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
		Step 1: Cache the training/test options
		Step 2: connect to a visdom server
		Step 3: create an HTML object for saveing HTML filters
		Step 4: create a logging file to store training losses
		"""
		self.opt = opt  # cache the option
		self.display_id = opt.display_id
		self.use_html = opt.phase == 'train' and not opt.no_html
		self.win_size = opt.display_winsize
		self.name = opt.name
		# self.port = opt.display_port
		self.saved = False
		self.plot_data = {'X': [], 'Y': [], 'Y val': [], 'legend': []}
		if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
			import visdom
			self.ncols = opt.display_ncols
			self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
			if not self.vis.check_connection():
				self.create_visdom_connections()
		if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
			self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
			self.img_dir = os.path.join(self.web_dir, 'images')
			print('create web directory %s...' % self.web_dir)
			util.mkdirs([self.web_dir, self.img_dir])
		# create a logging file to store training losses
		self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
		if os.path.isfile(self.log_name):
			with open(self.log_name, "a") as log_file:
				now = time.strftime("%c")
				log_file.write('================ Training Loss (%s) ================\n' % now)
	

	def reset(self):
		"""Reset the self.saved status"""
		self.saved = False

	def create_visdom_connections(self):
		"""If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
		cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
		print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
		print('Command: %s' % cmd)
		Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

	def get_losses(self, epoch, counter_ratio, losses):
		if not self.plot_data['legend']:
		  self.plot_data['legend'] = list(losses.keys())
		self.plot_data['X'].append(epoch + counter_ratio)
		self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
		
	def get_val_losses(self, losses):
		self.plot_data['Y val'].append([losses[k] for k in self.plot_data['legend']]) 
	  
	def plot_losses(self):
		fig = plt.figure()
		fig.suptitle('Gráfico de loss de treino', fontsize = 20)
		plt.xlabel('Época', fontsize = 18)
		plt.ylabel('Loss', fontsize = 16)
		axes = plt.gca()
		
		# axes.set_ylim([np.amin(self.plot_data['Y'][20:]), np.amax(self.plot_data['Y val'][20:])])
		plt.plot(self.plot_data['X'], self.plot_data['Y'], label = f"Y {self.plot_data['legend'][0]}")
		if self.plot_data['Y val']:
			plt.plot(self.plot_data['X'], self.plot_data['Y val'], label =  f"Y val {self.plot_data['legend'][0]}")
		plt.legend(loc="upper right")
		plt.savefig(os.path.join(self.img_dir, "all_losses.png"))
		
	def display_current_results(self, visuals, epoch, save_result):

		"""Display current results on visdom; save current results to an HTML file.

		Parameters:
			visuals (OrderedDict) - - dictionary of images to display or save
			epoch (int) - - the current epoch
			save_result (bool) - - if save the current results to an HTML file
		"""
		if self.display_id > 0:  # show images in the browser using visdom
			ncols = self.ncols
			if ncols > 0:        # show all the images in one visdom panel
				ncols = min(ncols, len(visuals))
				h, w = next(iter(visuals.values())).shape[:2]
				table_css = """<style>
						table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
						table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
						</style>""" % (w, h)  # create a table css
				# create a table of images.
				title = self.name
				label_html = ''
				label_html_row = ''
				images = []
				idx = 0
				for label, image in visuals.items():
					image_numpy = util.tensor2im(image)
					label_html_row += '<td>%s</td>' % label
					images.append(image_numpy.transpose([2, 0, 1]))
					idx += 1
					if idx % ncols == 0:
						label_html += '<tr>%s</tr>' % label_html_row
						label_html_row = ''
				white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
				while idx % ncols != 0:
					images.append(white_image)
					label_html_row += '<td></td>'
					idx += 1
				if label_html_row != '':
					label_html += '<tr>%s</tr>' % label_html_row
				try:
					self.vis.images(images, nrow=ncols, win=self.display_id + 1,
									padding=2, opts=dict(title=title + ' images'))
					label_html = '<table>%s</table>' % label_html
					self.vis.text(table_css + label_html, win=self.display_id + 2,
								  opts=dict(title=title + ' labels'))
				except VisdomExceptionBase:
					self.create_visdom_connections()

			else:     # show each image in a separate visdom panel;
				idx = 1
				try:
					for label, image in visuals.items():
						image_numpy = util.tensor2im(image)
						self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
									   win=self.display_id + idx)
						idx += 1
				except VisdomExceptionBase:
					self.create_visdom_connections()

		if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
			self.saved = True
			# save images to the disk
			for label, image in visuals.items():
				image_numpy = util.tensor2im(image)
				img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
				util.save_image(image_numpy, img_path)

			# update website
			webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, visuals, refresh=1)
			for n in range(epoch, 0, -1):
				webpage.add_header('epoch [%d]' % n)
				ims, txts, links = [], [], []

				for label, image_numpy in visuals.items():
					image_numpy = util.tensor2im(image)
					img_path = 'epoch%.3d_%s.png' % (n, label)
					ims.append(img_path)
					txts.append(label)
					links.append(img_path)
				webpage.add_images(ims, txts, links, width=self.win_size)
			webpage.save()

	def plot_current_losses(self, epoch, counter_ratio, losses):
		"""display the current losses on visdom display: dictionary of error labels and values

		Parameters:
			epoch (int)           -- current epoch
			counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
			losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
		"""
		if not hasattr(self, 'plot_data'):
			self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
		self.plot_data['X'].append(epoch + counter_ratio)
		self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
		try:
			self.vis.line(
				X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
				Y=np.array(self.plot_data['Y']),
				opts={
					'title': self.name + ' loss over time',
					'legend': self.plot_data['legend'],
					'xlabel': 'epoch',
					'ylabel': 'loss'},
				win=self.display_id)
		except VisdomExceptionBase:
			self.create_visdom_connections()

	# losses: same format as |losses| of plot_current_losses
	def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
		"""print current losses on console; also save the losses to the disk

		Parameters:
			epoch (int) -- current epoch
			iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
			losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
			t_comp (float) -- computational time per data point (normalized by batch_size)
			t_data (float) -- data loading time per data point (normalized by batch_size)
		"""
		message = '(epoch: %d, iters: %d, time: %.6f, data: %.6f) ' % (epoch, iters, t_comp, t_data)
		for k, v in losses.items():
			message += '%s: %.6f ' % (k, v)

		print(message)  # print the message
		with open(self.log_name, "a") as log_file:
			log_file.write('%s\n' % message)  # save the message
