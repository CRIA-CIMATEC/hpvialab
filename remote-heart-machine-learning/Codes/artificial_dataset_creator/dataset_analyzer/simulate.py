from scipy.interpolate import Akima1DInterpolator
from neurokit2.signal import signal_resample
from matplotlib import pyplot as plt
import heartpy as hp
import numpy as np
import scipy
import math

def ppg_simulate(pulse_signal, current_rate=256, desired_rate=30, random_state=None, show=False):
	"""Function that simulates a PPG based on other PPG or ECG signal.
	
	Keyword arguments
	-----------------
	pulse_signal : np.ndarray
		PPG or ECG that will be processed.

	current_rate : int
		Sampling rate of the input signal.

	desired_rate : int
		Desired sampling rate for the output signal.

	random_state : int
		Numpy random seed.

	show : bool
		Plot result.

	Return
	------ 
	ppg : np.array
		PPG array with `desired_rate` as sample rate.
	"""
	np.random.seed(random_state)

	wd, _ = hp.process(pulse_signal, sample_rate=current_rate)

	# get signal duration in seconds
	duration = np.ceil(len(pulse_signal) / current_rate)

	# get signal periods (IBI) based on the detected peaks
	intervals = np.array(wd['peaklist']) / current_rate
	intervals = np.diff(intervals)
	periods = np.array([*intervals, intervals[-1]]) # it must have the same number of peaks
	n_period = len(periods)

	# set the x position of the systolic peak based on the original signal peaks
	x_sys = np.array(wd['peaklist']) / current_rate
	
	# Corresponding signal amplitudes.
	y_onset = np.random.normal(0, 0.1, n_period)

	# get valleys of the wave
	x_onset = x_sys - np.random.normal(0.175, 0.01, n_period) * periods
	# Corresponding signal amplitudes.
	y_sys = np.array(wd['ybeat'])
	# resize the y peaks between 3.1 and 4 to get a cleanner signal
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
	
	if show:
		__, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
		ax0.scatter(x_all, y_all, c="r")

	# Interpolate a continuous signal between the landmarks (i.e., Cartesian
	# coordinates).
	f = Akima1DInterpolator(x_all, y_all)
	samples = np.arange(int(np.ceil(duration * desired_rate)))
	ppg = f(samples)
	# Remove NAN (values outside interpolation range, i.e., after last sample).
	ppg[np.isnan(ppg)] = np.nanmean(ppg)

	if show:
		ax0.plot(ppg)
		ax1.plot(ppg)

	return ppg

def ecg_simulate(pulse_signal=None, current_rate=256, desired_rate=256, ti=(-70, -15, 0, 15, 100), 
				ai=(1.2, -5, 30, -7.5, 0.75), bi=(0.25, 0.1, 0.1, 0.1, 0.4)):
	"""Function that simulates a ECG based on other PPG or ECG signal.

	This function is based on the `_ecg_simulate_ecgsyn` function from neurokit2:
	https://neurokit2.readthedocs.io/en/latest/_modules/neurokit2/ecg/ecg_simulate.html#ecg_simulate
	
	Keyword arguments
	-----------------
	pulse_signal : np.ndarray
		PPG or ECG that will be processed.

	current_rate : int
		Sampling rate of the input signal.

	desired_rate : int
		Desired sampling rate for the output signal.

	ti : tuple
		Parameters of the ECG specificities: angles of extrema (in degrees).

	ai : tuple
		Parameters of the ECG specificities: z-position of extrema.

	bi : tuple
		Parameters of the ECG specificities: Gaussian width of peaks.

	Return
	------ 
	ecg : np.array
		ECG array with `desired_rate` as sample rate.
	"""
	if not isinstance(ti, np.ndarray):
		ti = np.array(ti)
	if not isinstance(ai, np.ndarray):
		ai = np.array(ai)
	if not isinstance(bi, np.ndarray):
		bi = np.array(bi)

	wd, m = hp.process(pulse_signal, sample_rate=current_rate)

	out_ecg_len = (len(pulse_signal) * desired_rate) / current_rate 

	ti = ti * np.pi / 180

	# Adjust extrema parameters for mean heart rate
	hrfact = np.sqrt(m['bpm'] / 60)
	hrfact2 = np.sqrt(hrfact)
	bi = hrfact * bi
	ti = np.array([hrfact2, hrfact, 1, hrfact, hrfact2]) * ti

	# rr0 based o the input signal
	rr0 = wd['RR_list'] / 1000
	rr = signal_resample(rr0, sampling_rate=1, desired_sampling_rate=out_ecg_len//len(rr0))

	# Make the rrn time series
	dt = 1 / desired_rate
	rrn = np.zeros(len(rr))
	tecg = 0
	i = 0
	while i < len(rr):
		tecg += rr[i]
		ip = int(np.round(tecg / dt))
		rrn[i:ip] = rr[i]
		i = ip
	Nt = ip

	# Integrate system using fourth order Runge-Kutta
	x0 = np.array([1, 0, 0.04])

	# tspan is a tuple of (min, max) which defines the lower and upper bound of t in ODE
	# t_eval is the list of desired t points for ODE
	# in Matlab, ode45 can accepts both tspan and t_eval in one argument
	Tspan = [0, (Nt - 1) * dt]
	t_eval = np.linspace(0, (Nt - 1) * dt, Nt)

	# as passing extra arguments to derivative function is not supported yet in solve_ivp
	# lambda function is used to serve the purpose
	result = scipy.integrate.solve_ivp(
		lambda t, x: _ecg_simulate_derivsecgsyn(t, x, rrn, ti, desired_rate, ai, bi), Tspan, x0, t_eval=t_eval
	)
	X = result.y

	# Scale signal to lie between -0.4 and 1.2 mV
	z = X[2, :].copy()
	zmin = np.min(z)
	zmax = np.max(z)
	zrange = zmax - zmin
	z = (z - zmin) * 1.6 / zrange - 0.4

	return z  # Return signal

def _ecg_simulate_derivsecgsyn(t, x, rr, ti, desired_rate, ai, bi):
	"""This function is the `_ecg_simulate_derivsecgsyn` function from neurokit2:
	https://neurokit2.readthedocs.io/en/latest/_modules/neurokit2/ecg/ecg_simulate.html#ecg_simulate
	"""
	
	ta = math.atan2(x[1], x[0])
	r0 = 1
	a0 = 1.0 - np.sqrt(x[0] ** 2 + x[1] ** 2) / r0

	ip = np.floor(t * desired_rate).astype(int)
	w0 = 2 * np.pi / rr[min(ip, len(rr) - 1)]
	# w0 = 2*np.pi/rr[ip[ip <= np.max(rr)]]

	fresp = 0.25
	zbase = 0.005 * np.sin(2 * np.pi * fresp * t)

	dx1dt = a0 * x[0] - w0 * x[1]
	dx2dt = a0 * x[1] + w0 * x[0]

	# matlab rem and numpy rem are different
	# dti = np.remainder(ta - ti, 2*np.pi)
	dti = (ta - ti) - np.round((ta - ti) / 2 / np.pi) * 2 * np.pi
	dx3dt = -np.sum(ai * dti * np.exp(-0.5 * (dti / bi) ** 2)) - 1 * (x[2] - zbase)

	dxdt = np.array([dx1dt, dx2dt, dx3dt])
	return dxdt
