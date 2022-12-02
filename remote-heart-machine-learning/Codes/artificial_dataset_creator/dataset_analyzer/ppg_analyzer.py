from scipy.interpolate import interp1d
from scipy.io import loadmat, savemat
from matplotlib import pyplot as plt
from collections.abc import Iterable
import matplotlib.colors as mcolors
from scipy import stats
import heartpy as hp
import numpy as np
import natsort
import logging
import os

def array_flatten(array):
	"""Function that receives an array and returns its values without the nested lists.

	Keyword arguments
	-----------------
	array : Iterable
		Array that will be processed.

	Return
	------ 
	flatten : np.array
		Array after flattening.
	"""
	# Flatten based on NOROK2 response code: https://stackoverflow.com/questions/64099396/flatten-a-list-containing-numpy-arrays-with-different-shapes
	
	# Treats the data for a compatible format with Flatten that will be done:
	# Becomes Numpy Array if it is not
	if isinstance(array, np.ndarray) == False:
		array = np.array(array)

	values_list = []
	# Applies the Flatten
	# For each Numpy Array item
	for x in array.ravel():
		# Becomes Numpy Array if it is not
		if isinstance(x, np.ndarray) == False:
			x = np.array(x)

		# `.ravel()` applies a Flatten in nested values
		values_list.append(x.ravel())
	
	return np.concatenate(values_list)

def plot_multivalued(ppg_path: str, subject_name: str, subject_folder_path: str = None, save = False, show = True) -> None:
	"""Function that opens the PPG and plots the multivalorated fields along with their corrections.
	
	Notes
	-----
	- The PPG file must be in Matlab format and must contain the 'pulseOxRecord' field;
	- The subject name (`subject_name`) will be used in the plot title;
	- The path of the subject folder (`subject_folder_path`) will be used if the plot has to be saved;
	- The name of the image with the plot after saving the following structure: f'multivalued_ppg_{index}.png', \
		where the variable 'index' will mean the position of the multivalorated field in the array.
	
	Keyword arguments
	-----------------
	ppg_path : str
		Matlab file that must contain the 'pulseOxRecord' field
	
	subject_name : str
		Subject name will be used in plot title
	
	subject_folder_path : str
		Path of the subject's folder will be used if the plot has to be saved
	
	save : bool
		Boolean responsible for saving the plot or not
	
	show : bool
		Boolean responsible for showing plot or not
	"""

	if subject_folder_path:
		assert os.path.isdir(subject_folder_path), f'The `subject_folder_path` variable it\'s not a directory: {subject_folder_path}'
	
	if save:
		assert subject_folder_path, f'The `subject_folder_path` parameter should be a string when the save parameter its `True`: {subject_folder_path}'
	
	# loads the '.mat' file that should contain the field 'pulseOxRecord'
	mat = loadmat(ppg_path)

	assert 'pulseOxRecord' in mat.keys(), f'The PPG file path must have a tuple in dictionary with key called "pulseOxRecord" containing PPG data: {ppg_path}'

	# in the key 'pulseOxRecord' dictionary and in the first position (always exists before the given itself)
	ppg = mat['pulseOxRecord'][0]
	
	assert isinstance(ppg, Iterable), f'The value within the key "pulseOxRecord" In the PPG dictionary shall contain a `Iterable`. Type found: {type(ppg)}'
	
	# for each array inside the PPG:
	for index, array in enumerate(ppg):
		# if it does not contain only a value:
		if len(array_flatten(array.copy())) != 1:
			# Adds a space before and after Index to see correction and other wrists
			space = 50

			# Corrects the PPG in the position that contains several values
			ppg_rectified = np.concatenate([array.ravel() for array in ppg[index-space:index+space].ravel()])
	
			# Uses the media of PPG values without worrying about multivaluated fields (posing a varied values)
			values = np.array([int(np.mean(value)) for value in ppg[index-space:index+space]])
			
			plt.figure(figsize=(20, 10))
			# PPG without multivaled fields
			plt.plot(np.arange(index-space, index-space+len(ppg_rectified)), ppg_rectified, color='r', label='PPG without multivaled fields')
			# PPG with multivaled fields
			plt.plot(np.arange(index-space, index-space+space*2), values, color='b', label='PPG with multivaled fields')
			plt.title(f'Example of multivaluated field in the subject\'s PPG: {subject_name}')
			plt.legend()

			if show:
				plt.show(block=False)
			if save:
				plt.savefig(os.path.join(subject_folder_path, f'multivalued_ppg_{index}.png'))

def plot_measure_bars(measures_dict: dict, use_measures: list, title: str, out_path=None, 
					out_filename=None, fig_size: tuple=(20, 10), save=False, show=True) -> None:
	"""Function that creates a bar plot with the measurements of each subject.
	
	Notes
	-----
	- The parameter `out_path` should not be null when the save parameter is `True`.
	- The `out_filename` parameter should not be null when the save parameter is `True`.
	- The `out_path` parameter must be an existing directory.
	
	Keyword arguments
	-----------------
	measures_dict : dict
		Dictionary that contemplates the names of the measures such as the keys and values of the measures such as dict values.
	
	use_measures : list
		List of names of measurements that should be used in the plot. Names must match the `measures_dict` parameter keys.
	
	title : str
		Chart title that will be ploted.
	
	out_path : str
		Path of the folder used to save the plot if the `save` parameter is `True`.
	
	out_filename : str
		File name (without extension) used to save the plot if the `save` parameter is` true`.
	
	fig_size : tuple
		Graphic size that will be plotted.
	
	save : bool
		Boolean responsible for saving the plot or not.
	
	show : bool
		Boolean responsible for showing plot or not.
	"""
	# Checks the function conditions
	if save:
		assert out_path, f"The parameter `out_path` of the function `plot_measure_bars` should not be null when the `save` parameter is `True`: {out_path}"
		assert os.path.isdir(out_path), f"The `out_path` parameter of the `plot_measure_bars` function should be an existing directory: {out_path}"
		assert out_filename, f"The 'out_filename' parameter of the function `plot_measure_bars' should not be null when the `save` parameter is `True`: {out_filename}"
	
	plt.figure(figsize=fig_size)

	# Collecting color list to use on the bars
	color = list(mcolors.BASE_COLORS.keys()) + list(mcolors.TABLEAU_COLORS.keys())
	# removes the white color from the list to not get invisible in the plot
	color.remove('w')

	# for each `measures_dict`: 
	# key: name of the subject
	# value: Measurement Dictionary
	for i, (key, value) in enumerate(measures_dict.items()):
		# Filter measures the user want to show
		measures = {nome: measure for nome, measure in value.items() if nome in use_measures}

		# The plot will have len(measures) clusters
		positions = np.array(range(len(measures)))
		width = 0.5 / len(measures_dict)

		plt.bar(
			x=positions + ((i/2)/len(measures_dict)), 
			height=measures.values(), 
			label=key, 
			color=color[i], 
			width=width
		)

	plt.xticks(positions + width, measures.keys())
	plt.title(title)
	plt.ylabel('Measure value')
	plt.xlabel('Measures (heartpy)')
	plt.legend()

	if save:
		plt.savefig(os.path.join(out_path, f'{out_filename}.png'))
	if show:
		plt.show()

def interpolate_grow(bpm: Iterable, stride: int or float):
	"""Function that applies linear interpolation between the values of a time series, makes it grow `stride` times.
	
	Notes
	-----
	- The `bpm` array should not contain NAN, use the `interpolate_nan` function before that one.
	- The function will make a linear interpolation.
	
	Keyword arguments
	-----------------
	bpm : Iterable
		Time series in which you want to apply the interpolation between your points.
	
	stride : int or float
		Numeric value that will point in how many times the time series will grow.

	Return
	------
	bpm : Iterable
		Time series as after interpolation.
	"""
	bpm = np.array(bpm)
	
	assert np.isnan(bpm).any() == False, "The BPM array should not contain NAN, use the `interpolate_nan` function before this"

	# Creates x for each position:
	# >>> x = array([4, 8, 12, 16, 20, ...])
	# len(x) igual a len(measures)
	x = np.arange(start=stride, stop=len(bpm)*stride+1, step=stride)

	interp = interp1d(x, bpm, kind='linear', bounds_error=False)

	# Applies interpolation in the range that has four times the size of the `bpm` variable:
	# >>> array(0, 1, 2, 3, ...)
	bpm = interp(np.arange(start=stride, stop=len(bpm)*stride+1))
	
	return bpm

def interpolate_outliers(bpm):
	"""Function that applies linear interpolation in the outliers of the time series.
	
	Notes
	-----
	- The `bpm` array should not contain NAN, use the `interpolate_nan` function before that one.
	- The function will make a linear interpolation of the points that exceed 3 of Z-score.
	
	Keyword arguments
	-----------------
	bpm : Iterable
		Time series in which you want to apply the interpolation of your outliers.

	Return
	------
	bpm : Iterable
		Time series as after the outliers interpolation.
	"""
	# Z-score based code: https://www.geeksforgeeks.org/detect-and-remove-the-outliers-using-python/
	bpm = np.array(bpm)
	
	assert np.isnan(bpm).any() == False, "The BPM array should not contain NAN, use the `interpolate_nan` function before this"
	
	# calculates the z-score of each point to determine outliers
	z = np.abs(stats.zscore(bpm))

	x = np.arange(len(bpm))
	
	# Collect `bpm` without outliers
	bpm_filtered = bpm[np.where(z < 3)[0]]
	x = x[np.where(z < 3)[0]]
	
	if x != []:
		interp = interp1d(x, bpm_filtered, kind='linear', bounds_error=False)

		# Applies interpolation in the range with the total size of variable BPM
		bpm = interp(np.arange(len(bpm)))
	
	return bpm

def interpolate_nan(bpm):
	"""Function that applies linear interpolation in the NAN values of the time series.
	
	Notes
	-----
	- The function will make a linear interpolation of the points that are NAN;
	- The first value will be repeated if the first position contains a NAN;
	- The last value will be repeated if the last position contains a NAN.
	
	Keyword arguments
	-----------------
	bpm : Iterable
		Time series in which you want to apply the interpolation of your NAN values.

	Return
	------
	bpm : Iterable
		Time series as after the NAN interpolation.
	"""
	bpm = np.array(bpm)

	x = np.arange(len(bpm))

	# The first value will be repeated if the first position contains a NAN.
	if np.isnan(bpm[0]):
		bpm[0] = bpm[np.where(np.isnan(bpm) == False)[0]].mean()

	# The last value will be repeated if the last position contains a NAN.
	if np.isnan(bpm[-1]):
		bpm[-1] = bpm[np.where(np.isnan(bpm) == False)[0]].mean()

	# Collect `bpm` without NAN Values
	bpm_filtered = bpm[np.where(np.isnan(bpm) == False)[0]]
	x = x[np.where(np.isnan(bpm) == False)[0]]

	interp = interp1d(x, bpm_filtered, kind='linear', bounds_error=False)

	# Applies interpolation in the range with the total size of variable BPM
	bpm = interp(np.arange(len(bpm)))
	
	return bpm

def explore_ppg(dataset_path: str, ppg_fps: int=60, segment_width=4, strides: Iterable=[4], save_mat=False, save_plots=False, show=True) -> None:
	"""Function that opens each PPG uses the Heartpy library to calculate and plot statistics.
	
	Notes
	-----
	- The PPG file in Matlab format must be immediately within each subject's folder
	- The `segment_overlap` parameter must be larger than zero
	- The `stride` parameter should be larger than zero
	- Each value of the `stride` parameter has to be smaller than the `segment_overlap`
	- Calculation of overlapping grade of windows:
	   
	>>> segment_overlap = (segment_width - stride) * (1 / segment_width) # where:
	>>> (1) 0 < segment_width >= stride
	>>> (2) 0 <= segment_overlap < 1

	Keyword arguments
	-----------------
	dataset_path : str
		Dataset paste path containing subjects as subfolders
	
	ppg_fps : int or real
		Sampling rate that the PPG capture device used
	
	segment_width : int or real
		Window size (in seconds) that will be applied on PPG
	
	strides : Iterable
		List that contains strides that should be applied in the `heartpy.process_segmentwise` function.
		Each stride will determine the degree of overlapping that the windows will have with each other.
		Example of windows:
		>>> # 1234 5678 ...         (stride: 4)
		>>> # 1234 2345 3456 ...    (stride: 1)
	
	save_mat : bool
		Boolean responsible for determining if the Matlab file will be saved with another name or not
	
	save_plots : bool
		Boolean responsible for determining whether the plots will be saved in the dataset folders or not
	
	show : bool
		Boolean responsible for determining if the graphics or not.
	"""
	# Verification of folders and base files

	# folder of the base dataset has to exist
	assert os.path.isdir(dataset_path), f"The `dataset_path` parameter isn't a directory: {dataset_path}"
	assert segment_width > 0, f"The `segment_width` parameter should be larger than zero seconds: {segment_width}"
	assert (segment_width >= array_flatten(strides.copy())).all(), f"One of the values of the `strides` parameter is larger than the `segment_width` parameter ({segment_width}): \n{strides}"

	ppg_paths = []
	# Collecting color list to use on the bars
	color = list(mcolors.BASE_COLORS.keys()) + list(mcolors.TABLEAU_COLORS.keys())
	# removes the white color from the list to not get invisible in the plot
	color.remove('w')
	color.remove('g')

	# For each folder within the base dataset
	for subject_folder_name in os.listdir(dataset_path):
		subject_folder = os.path.join(dataset_path, subject_folder_name)

		# To assume that the `subject_folder` folder may contain a subject we have to verify:
		# if it is a directory
		# If it contains the 'PulseOx' folder and the 'PulseOx.mat' file exists
		# If it contains one of the following folders: RGB or NIR
		if os.path.exists(os.path.join(subject_folder, 'pulseOx.mat')):
			ppg_paths.append((subject_folder_name, subject_folder, os.path.join(subject_folder, 'pulseOx.mat')))

	# Order naturally to Path List
	ppg_paths = natsort.natsorted(ppg_paths)

	dataset_json = {}
	bpm_hist = {}

	# for each video that needs to be processed
	for (subject_folder_name, subject_folder_path, ppg_path) in ppg_paths:
		try:
			# loads the '.mat' file that should contain the 'PulseOxRecord' field
			mat = loadmat(ppg_path)
		   
			ppg = mat['pulseOxRecord'][0]

			# ppg = np.concatenate([x.ravel() for x in pulse.ravel()])

			wd, m = hp.process(ppg, sample_rate=ppg_fps)

			pointcare = hp.plot_poincare(wd, m, show=False, figsize=(20, 10))
			if save_plots:
				pointcare.savefig(os.path.join(subject_folder_path, 'pointcare.png'))
			if show:
				pointcare.show()

			hp.plotter(wd, m, figsize=(20, 10))
 
			# Adds in the dictionary the measurements calculated by the Heartpy library
			dataset_json[subject_folder_name] = m

			# for each stride
			for i, stride in enumerate(strides):
				# Process PPG to get BPM
				m['bpm'] = ppg_to_bpm(ppg, ppg_fps=ppg_fps, stride=stride, segment_width=segment_width)

				plt.plot(np.arange(len(m['bpm'])), m['bpm'], color=color[i], label=f'Heart rate - Stride: {stride} seconds')

			# Adds the beats collected in the last loop to plot the histogram
			bpm_hist[subject_folder_name] = m['bpm']
			
			plt.title(f'Peaks of ppg pulses and beats per minute - {subject_folder_name}')
			plt.ylabel('Beats per minute (BPM)')
			plt.xlabel('Time (seconds)')
			plt.legend()
			if show:
				plt.show()
			if save_plots:
				plt.savefig(os.path.join(subject_folder_path, 'ppg_hr_graph.png'))
			
			if save_mat:
				# Creates a new key to save the Matlab file
				mat['bpm'] = m['bpm']

				savemat(os.path.join(subject_folder_path, 'pulseOx_bpm.mat'), mat)
		except Exception as e:
			print(f"Unmapped Exception: \n{logging.traceback.format_exc()}")
			pass

	if bpm_hist != {}:
		plt.figure(figsize=(20, 10))
		# It generates the histogram with the beats of the subjects
		for subject, bpms in bpm_hist.items():
			plt.hist(bpms, label=subject, alpha=0.5)
		plt.legend()
		plt.title('Histogram of the beating distribution of each subject')
		plt.ylabel('Frequency')
		plt.xlabel('Beats per minute (BPM)')
		if save_plots:
			plt.savefig(os.path.join(dataset_path, 'bpm_histogram.png'))
		if show:
			plt.show()

	if dataset_json != {}:
		# creates parameters to use in the loop below
		params = [
			{
				'out_filename': 'measures_1', 'title': 'Time-domain Measurements',
				'use_measures': ['bpm', 'sdnn', 'sdsd', 'mssd', 'hr_mad', 'sd1', 'sd2'], 
			},
			{
				'out_filename': 'measures_2', 'title': 'Medium interval between beats (milliseconds)', 
				'use_measures': ['ibi'], 
			},
			{
				'out_filename': 'measures_3', 'title': 'Time-domain Measurements', 
				'use_measures': ['pnn20', 'pnn50', 'sd1/sd2', 'breathingrate'], 
			},
		]

		# For each dictionary of the parameters:
		for param in params:
			# Pass the standard parameters and parameters stop as **kwargs
			plot_measure_bars(measures_dict=dataset_json, out_path=dataset_path, save=save_plots, show=show, **param)

def ppg_to_bpm(ppg, ppg_fps: int=30, stride=4, segment_width=4):
	"""Function that produces the BPM in each second of the PPG or ECG signal.
	
	Keyword arguments
	-----------------
	ppg : np.ndarray
		PPG or ECG that will be processed.

	ppg_fps : int
		Sampling rate of the signal.

	stride : int
		Step of the sliding window that produces BPM.

	segment_width : int
		Window size that produces BPM.

	Return
	------ 
	bpm : np.array
		BPM array.
	"""

	# if the video lenght (seconds) is bigger than two times the segment_width (segment_width*2):
    #   segment_width = segment_width (parameter)
    # else:
    #   segment_width = 1 # because the ppg is too short
	if (len(ppg) / ppg_fps) <= segment_width*2:
		segment_width = 1
		stride = 1

	# Calculation of the overlapping grade of the windows:
	# segment_overlap = (segment_width - stride) * (1 / segment_width), where:
	# (1) 0 < segment_width >= stride
	# (2) 0 <= segment_overlap < 1
	segment_overlap = (segment_width - stride) * (1 / segment_width)

	# the 'fast' mode has better results, but it does not always work
	try:
		_, m = hp.process_segmentwise(
			ppg, 
			sample_rate=ppg_fps, 
			segment_width=segment_width, 
			segment_overlap=segment_overlap, 
			mode='fast'
		)
	except:
		_, m = hp.process_segmentwise(
			ppg, 
			sample_rate=ppg_fps, 
			segment_width=segment_width, 
			segment_overlap=segment_overlap, 
			mode='full'
		)

	# Applies pre-processing in beat values per minute:
	# interpolation of Nan values
	m['bpm'] = interpolate_nan(m['bpm'])
	if stride > 1:
	# interpolation in the values to maintain the quantity of seconds
		m['bpm'] = interpolate_grow(m['bpm'], stride=stride)
	# Interpolation of outliers values 
	m['bpm'] = interpolate_outliers(m['bpm'])  

	# the first value in the first second seconds that were not contemplated because of the winding
	m['bpm'] = [*[m['bpm'][0] for j in range(3)], *m['bpm']]

	# calculates the amount of beating missing at the end of the series
	quant_bpm_end = int(len(ppg) / ppg_fps) - len(m['bpm'])

	# Admission of the last value in the seconds that are missing because of the window
	m['bpm'] = [*m['bpm'], *[m['bpm'][-1] for j in range(quant_bpm_end)]]

	# interpolation of Nan values after all process
	m['bpm'] = interpolate_nan(m['bpm'])

	return m['bpm']
