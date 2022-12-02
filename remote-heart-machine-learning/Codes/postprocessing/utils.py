import sys
sys.path.append('../artificial_dataset_creator')
from dataset_analyzer.ppg_analyzer import interpolate_nan

from scipy import stats
from scipy.interpolate.interpolate import interp1d

from scipy.interpolate import Akima1DInterpolator
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.signal import medfilt
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
import heartpy
import neurokit2
import math

def postprocessing(pulse_signal, current_rate=256, desired_rate=30, remove_rejected=False, random_state=None):
	"""Function that apply the postprocessing to the PPG. The result is better if the PPG is clean. You could \
		use the `neurokit2.ppg_clean` function to clean it before apply the post processing.

	Keyword arguments
	-----------------
	pulse_signal : np.ndarray
		PPG that needs to be processed.

	current_rate : int
		Current sample rate of the `pulse_signal` PPG.

	desired_rate : int
		Desired output sample rate.

	remove_rejected : bool
		Wheter the postprocessing should remove the rejected peaks or use it.

	random_state : int
		Numpy random state.

	Return
	------
	(ppg, measures) : tuple
		ppg : np.ndarray
			Post processed PPG.
		measures : dict
			Dict with some standard deviation and variance about the periods.
	"""
	np.random.seed(random_state)

	duration = np.ceil(len(pulse_signal) / current_rate)
	# get valid peaks and valleys
	peaks, _ = get_peaks_valleys(pulse_signal, current_rate, remove_rejected=remove_rejected)

	# get the periods to calculate the std and var from the periods before treat the PPG.
	x_onset = np.asarray(peaks[:, 0]) / current_rate
	periods = np.diff(x_onset)
	periods_old = np.array([*periods, periods[-1]])

	# fix the holes or the close pulses of the PPG
	fixed_peaks, periods = fix_gaps(peaks, current_rate, duration)

	# applies a median filter to supress hard changes
	periods = medfilt(periods, 9) # best >= 9
	
	# get the higher points of the PPG
	y_sys = np.asarray(fixed_peaks[:, 1])

	# makes the periods to have the correct duration
	periods += ((duration - periods.sum()) / len(periods))

	measures = {
		'Original_rr_sd': np.std(periods_old),
		'Original_rr_var': np.var(periods_old),
		'Simulated_rr_sd': np.std(periods),
		'Simulated_rr_var': np.var(periods),
		'periods': periods
	}

	n_period = len(periods)

	# the periods are the interval between the higher points of the PPG
	x_sys = np.cumsum(periods)
	x_sys[0] = x_onset[0]

	# lower point of the PPG
	y_onset = np.random.normal(0, 0.1, n_period)

	# Seconds at which the systolic peaks occur within the waves.
	x_onset = x_sys - np.random.normal(0.175, 0.01, n_period) * periods
	# Rescale the higher points of the wave between 3.1 and 4
	if y_sys.min() < 3.1 or y_sys.max() > 4:
		y_sys = np.interp(y_sys, (y_sys.min(), y_sys.max()), (3.1, 4))

	# Seconds at which the dicrotic notches occur within the waves.
	x_notch = x_onset + np.random.normal(0.4, 0.001, n_period) * periods
	# Corresponding signal amplitudes (percentage of systolic peak height).
	y_notch = y_sys * np.random.normal(0.49, 0.01, n_period)

	# Seconds at which the diastolic peaks occur within the waves.
	x_dia = x_onset + np.random.normal(0.45, 0.001, n_period) * periods
	# Corresponding signal amplitudes (percentage of systolic peak height).
	y_dia = y_sys * np.random.normal(0.47, 0.01, n_period)

	x_all = np.concatenate((x_onset, x_sys, x_notch, x_dia))
	x_all.sort(kind="mergesort")
	x_all = x_all * desired_rate  # convert seconds to samples

	y_all = np.zeros(n_period * 4)
	y_all[0::4] = y_onset
	y_all[1::4] = y_sys
	y_all[2::4] = y_notch
	y_all[3::4] = y_dia

	# Interpolate a continuous signal between the landmarks (i.e., Cartesian
	# coordinates).
	f = Akima1DInterpolator(x_all, y_all)
	samples = np.arange(int(np.ceil(duration * desired_rate)))
	ppg = f(samples)
	# Remove NAN (values outside interpolation range, i.e., after last sample).
	ppg[np.isnan(ppg)] = np.nanmean(ppg)
	
	# clears it after post processing
	ppg = neurokit2.ppg_clean(ppg, sampling_rate=current_rate)

	return ppg, measures

def fix_gaps(peaks, current_rate, duration):
	"""Function that fix the holes or the close pulses of the PPG.

	Keyword arguments
	-----------------
	peaks : np.ndarray
		Valid peaks that needs to be processed (x and y).

	current_rate : int
		Current sample rate of the `pulse_signal` PPG.

	duration : int
		Duration of the PPG (seconds).

	Return
	------
	(peaks, periods) : tuple
		peaks : np.ndarray
			Peaks after fix the gaps between then.
		periods : dict
			Interpolated periods.
	"""
	periods = get_periods(peaks[:, 0], current_rate)
	# makes the periods to have the correct duration
	periods += ((duration - periods.sum()) / len(periods))

	x_periods = np.arange(len(periods))
	x_peaks = np.arange(len(peaks[:, 0]))

	# calculates the z-score of each point to determine outliers
	z = np.abs(stats.zscore(periods))
	max_period = np.quantile(periods, 0.9)
	min_period = np.quantile(periods, 0.1)

	# set the max and min
	max_period = 1.5 if max_period > 1.5 else max_period
	min_period = 0.25 if min_period < 0.25 else min_period
	z_thr = 1 # 3 or 1
	
	# Collect `periods` without outliers
	periods_idxs = np.where(np.bitwise_and(np.bitwise_and(z < z_thr, periods >= min_period), periods <= max_period))[0]
	periods_filtered = periods[periods_idxs]
	x_periods = x_periods[periods_idxs]

	outlier_idxs = np.where(np.bitwise_or(np.bitwise_or(z >= z_thr, periods < min_period), periods > max_period))[0]

	quant_interp_periods = 0
	quant_interp_peaks = 0

	for i in range(len(outlier_idxs)):
		outlier_idx = outlier_idxs[i]
		before_outlier = 0
		after_outlier = 0

		if outlier_idx >= 1:
			before_outlier = periods[outlier_idx - 1]
		if outlier_idx + 1 < len(periods):
			after_outlier = periods[outlier_idx + 1]

		shift_interp = 0
		if periods[outlier_idx] > max_period or periods[outlier_idx] > periods_filtered.mean():
			# treat gaps adding new pulses inside
			if before_outlier == 0:
				shift_interp = round(periods[outlier_idx] / after_outlier) # math.ceil or round
			elif after_outlier == 0:
				shift_interp = round(periods[outlier_idx] / before_outlier) # math.ceil or round
			else:
				shift_interp = round(periods[outlier_idx] / periods.mean()) # math.ceil or round
		elif periods[outlier_idx] < min_period or periods[outlier_idx] < periods_filtered.mean():
			# treat close pulses adding time to periods from near periods
			if len(np.where(periods_idxs < outlier_idx)[0]) > 0:
				period_diff = min_period - periods[outlier_idx]
				dist_quant = int(period_diff // 0.1)
				dist_quant = 1 if dist_quant == 0 else dist_quant
				dist_before = math.ceil(dist_quant / 2)
				dist_after = dist_quant - dist_before
				last_period = np.where(periods_idxs < outlier_idx)[0][-1]
				periods_filtered[last_period-dist_before:last_period+dist_after] -= (period_diff / dist_quant)
				shift_interp = 1
		else:
			print(max_period, periods[outlier_idx])

		# give space to the periods and peaks to future interpolation
		x_periods[x_periods > outlier_idx + quant_interp_periods] += (shift_interp - 1)
		x_peaks[np.where(x_peaks > outlier_idx + quant_interp_peaks)[0]] += (shift_interp - 1)

		quant_interp_periods += shift_interp
		quant_interp_peaks += (shift_interp - 1)

	interp_periods = interp1d(x_periods, periods_filtered, kind='linear', bounds_error=False)

	interp_peaks_x = interp1d(x_peaks, peaks[:, 0], kind='linear', bounds_error=False)
	interp_peaks_y = interp1d(x_peaks, peaks[:, 1], kind='linear', bounds_error=False)
	
	# Applies interpolation in the range with the total size of variable periods
	periods = interp_periods(np.arange(len(periods_filtered) + quant_interp_periods))
	periods = interpolate_nan(periods)

	fixed_peaks_x = interp_peaks_x(np.arange(len(peaks) + quant_interp_peaks))
	fixed_peaks_y = interp_peaks_y(np.arange(len(peaks) + quant_interp_peaks))
	fixed_peaks_x = interpolate_nan(fixed_peaks_x)
	fixed_peaks_y = interpolate_nan(fixed_peaks_y)
	peaks = np.array([[x, y] for x, y in zip(fixed_peaks_x, fixed_peaks_y)])

	return peaks, periods

def get_periods(x_peaks, current_rate):
	"""Default method to get the periods.

	Keyword arguments
	-----------------
	x_peaks : np.ndarray
		Valid peaks that needs to be processed (only x).

	current_rate : int
		Current sample rate of the `pulse_signal` PPG.

	Return
	------
	periods : np.ndarray
		Periods from x_peaks.
	"""
	x_onset = np.asarray(x_peaks) / current_rate
	periods = np.diff(x_onset)
	return np.array([*periods, periods[-1]])

def get_valleys(peaks, valleys):
	"""Default method to get valid Valleys.

	Keyword arguments
	-----------------
	peaks : np.ndarray
		Valid peaks that will help to get the valleys(x and y).

	valleys : np.ndarray
		Valid valleys that needs to be processed (x and y).

	Return
	------
	fixed_valleys : np.ndarray
		Valleys (x and y).
	"""
	fixed_valleys = []
	# check every peaks to get the valleys
	for peak_t0, peak_t1 in zip(peaks[:-1], peaks[1:]):
		# get all valleys that are between the peaks
		detected_valleys = valleys[np.where((valleys[:, 0] > peak_t0[0]) & (valleys[:, 0] < peak_t1[0]))[0]]
		# filter valleys that are lower than the peaks
		detected_valleys = detected_valleys[np.where(
			(detected_valleys[:, 1] < peak_t0[1]) & 
			(detected_valleys[:, 1] < peak_t1[1])
		)[0]]
		# get lower valley
		min_valley = np.where(detected_valleys[:, 1] == min(detected_valleys[:, 1], default=None))[0]
		if min_valley.size > 0:
			fixed_valleys.append(detected_valleys[min_valley[0], :])
	
	fixed_valleys = np.array(fixed_valleys)
	# if there is a valley before a peak in the X axis
	if peaks[0, 0] > valleys[0, 0]:
		before_peak = valleys[np.where(valleys[:, 0] < peaks[0, 0])[0]]
		min_valley = np.where(before_peak[:, 1] == min(before_peak[:, 1], default=None))[0]
		if min_valley.size > 0:
			min_valley = before_peak[min_valley[0], :]
			# if the min_valley is not equal to fixed_valleys[-1, :] (or one is not the other)
			if (min_valley == fixed_valleys[-1, :]).all() == False:
				fixed_valleys = np.array([min_valley, *fixed_valleys])

	return fixed_valleys

def get_peaks(peaks, valleys):
	"""Default method to get the valid Peaks.

	Keyword arguments
	-----------------
	peaks : np.ndarray
		Valid peaks that needs to be processed (x and y).

	valleys : np.ndarray
		Valid valleys that will help to get the valleys (x and y).

	Return
	------
	fixed_peaks : np.ndarray
		Peaks (x and y).
	"""
	fixed_peaks = []
	# check every valleys to get the peaks
	for valley_t0, valley_t1 in zip(valleys[:-1, 0], valleys[1:, 0]):
		# get all peaks that are between the valleys
		detected_peaks = peaks[np.where((peaks[:, 0] > valley_t0) & (peaks[:, 0] < valley_t1))[0]]
		# get max peak
		max_peak = np.where(detected_peaks[:, 1] == max(detected_peaks[:, 1], default=None))[0]
		if max_peak.size > 0:
			fixed_peaks.append(detected_peaks[max_peak[0], :])

	fixed_peaks = np.array(fixed_peaks)

	first_peaks = peaks[np.where(
		(peaks[:, 0] < valleys[0, 0]) & 
		(peaks[:, 1] > valleys[0, 1])
	)[0]]

	if first_peaks.size > 0:
		first_peak = first_peaks[np.argmax(first_peaks[:, 1])]
		if (first_peak == fixed_peaks[0]).all() == False:
			fixed_peaks = np.array([first_peak, *fixed_peaks])

	# if there is a peak after the last valley in the X axis and
	# the last fixed_peak is not equal to peaks[-1,:] (or one is not the other)
	last_peaks = peaks[np.where(
		(peaks[:, 0] > valleys[-1, 0]) & 
		(peaks[:, 1] > valleys[-1, 1])
	)[0]]

	if last_peaks.size > 0:
		# last_peak = last_peaks[np.where(last_peaks[:, 1] == max(last_peaks[:, 1]))[0]]
		last_peak = last_peaks[np.argmax(last_peaks[:, 1])]
		if (last_peak == fixed_peaks[-1]).all() == False:
			fixed_peaks = np.array([*fixed_peaks, last_peak])

	return fixed_peaks

def get_peaks_valleys(ppg, sample_rate, remove_rejected=False):
	"""Default method to get valid Valleys and Peaks.

	Keyword arguments
	-----------------
	ppg : np.ndarray
		PPG to get the valid peaks and valleys.

	sample_rate : int
		Current sample rate of the `pulse_signal` PPG.

	remove_rejected : bool
		Wheter the postprocessing should remove the rejected peaks or use it.

	Return
	------
	(fixed_peaks, fixed_valleys) : tuple
		fixed_peaks : np.ndarray
			Fixed peaks (x and y).
		fixed_valleys : np.ndarray
			Fixed valleys (x and y).
	"""
	wd, _ = heartpy.process(ppg, sample_rate=sample_rate)
	ma_perc = wd['best']
	peaks = np.array(wd['peaklist'])

	if remove_rejected:
		# mask to remove rejected peaks
		mask = np.in1d(peaks, wd['removed_beats'], assume_unique=True, invert=True)
		peaks = peaks[mask]

	valleylist, _ = detect_valleys(ppg, wd['rolling_mean'], ma_perc, min_diff=0.9)

	peaks = np.array([[x, y] for x, y in zip(peaks, ppg[peaks])])
	valleys = np.array([[x, y] for x, y in zip(valleylist, ppg[valleylist])])
	
	fixed_valleys = get_valleys(peaks, valleys)
	fixed_peaks = get_peaks(peaks, fixed_valleys)

	return fixed_peaks, fixed_valleys

def plot(*arrays, config=[], title='', x_label='', y_label='', x_ticks=None, show=True):
	"""Common method to plot lines.
	
	Keyword arguments
	-----------------
		*arrays : list of arrays
			Lines to plots.
		
		config : list of dicts
			List of each matplotlib.pyplot config .

		title : str
			Common title that will appear at every plot.

		x_label : str
			Label of the x axis.

		y_label : str
			Label of the y axis.

		x_ticks : np.ndarray
			Ticks to insert on the x axis.

		show : bool
			Wheter to show the plot or not.

	Return
	------
	None
	"""
	
	configs = np.full(len(arrays), [{}])
	for i, user_conf in enumerate(config):
		configs[i] = user_conf

	for array, conf in zip(arrays, configs):
		plt.plot(array, **conf)
	
	plt.xlabel(x_label)
	plt.ylabel(y_label)

	if x_ticks is not None:
		locs = np.arange(0, len(arrays[0]), len(arrays[0])//len(x_ticks))
		plt.xticks(locs, labels=x_ticks)

		for i, tick in enumerate(plt.xticks()[1]):
			if i % 5 != 0:
				tick.set_visible(False)

	plt.title(title)
	plt.legend()
	if show:
		plt.show()

def mean_absolute_percentage_error(y, y_pred): 
	"""Method that calculates the MAPE"""
	if 0 in y:
		idxs = np.where(y == 0)[0]
		y = np.delete(y, idxs)
		y_pred = np.delete(y_pred, idxs)
		
	return np.mean(np.abs((y_pred - y) / y)) * 100

def mean_error(y, y_pred):
	"""Method that calculates the ME"""
	return np.mean(y_pred - y)
	
def stand_dev(y, y_pred):
	"""Method that calculates the standard deviation of the pred"""
	return np.std(y_pred)
  
def rmse(y, y_pred):
	"""Method that calculates the RMSE"""
	return mean_squared_error(y, y_pred, squared=False)

def pearson(y, y_pred):
	"""Method that calculates Pearson"""
	return pearsonr(y, y_pred)[0]
  
def mae(y, y_pred):
	"""Method that calculates the MAE"""
	return mean_absolute_error(y, y_pred)

def detect_valleys(hrdata, rol_mean, ma_perc, min_diff=1, update_dict=False, working_data={}):
	'''detect valleys in signal

	Function that detects heartrate valleys in the given dataset.

	Parameters
	----------
	hr data : 1-d numpy array or list
		array or list containing the heart rate data

	rol_mean : 1-d numpy array
		array containing the rolling mean of the heart rate signal

	ma_perc : int or float
		the percentage with which to raise the rolling mean,
		used for fitting detection solutions to data

	sample_rate : int or float
		the sample rate of the provided data set

	update_dict : bool
		whether to update the valley information in the module's data structure
		Settable to False to allow this function to be re-used for example by
		the breath analysis module.
		default : True

	Examples
	--------
	Normally part of the valley detection pipeline. Given the first example data
	it would work like this:

	>>> import heartpy as hp
	>>> from heartpy.datautils import rolling_mean, _sliding_window
	>>> data, _ = hp.load_exampledata(0)
	>>> rol_mean = rolling_mean(data, windowsize = 0.75, sample_rate = 100.0)
	>>> wd = detect_valleys(data, rol_mean, ma_perc = 20, sample_rate = 100.0)

	Now the valleylist has been appended to the working data dict. Let's look
	at the first five valley positions:

	>>> wd['valleylist'][0:5]
	[63, 165, 264, 360, 460]
	'''
	rmean = np.array(rol_mean)

	mn = np.mean(rmean / 100) * ma_perc
	rol_mean = rmean + mn

	valleysx = np.where((hrdata < rol_mean))[0]
	valleysy = hrdata[valleysx]
	valleyedges = np.concatenate((np.array([0]),
								(np.where(np.diff(valleysx) > min_diff)[0]), # np.diff(valleysx) > 1
								np.array([len(valleysx)])))
	valleylist = []

	for i in range(0, len(valleyedges)-1):
		try:
			y_values = valleysy[valleyedges[i]:valleyedges[i+1]].tolist()
			valleylist.append(valleysx[valleyedges[i] + y_values.index(min(y_values))])
		except:
			pass

	if update_dict:
		working_data['valleylist'] = valleylist
		return working_data
	else:
		return valleylist, working_data

