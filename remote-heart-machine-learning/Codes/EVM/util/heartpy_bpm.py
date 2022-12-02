import heartpy as hp
from collections.abc import Iterable
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from scipy import stats

def window_average(x, w):
    return np.convolve(x, np.ones(w), 'valid')[: : w] / w

def get_heartpy_bpm(ppg, ppg_fps, segment_width, stride, segmentwise=False):
    # Calculation of the overlapping grade of the windows:
    # segment_overlap = (segment_width - stride) * (1 / segment_width), where:
    # (1) 0 < segment_width >= stride
    # (2) 0 <= segment_overlap < 1
    segment_overlap = (segment_width - stride) * (1 / segment_width)

    if segmentwise:
        wd, m = hp.process_segmentwise(ppg, sample_rate=ppg_fps, segment_width=segment_width, segment_overlap=segment_overlap)
        # Applies pre-processing in beat values per minute:
        # Interpolacao of Nan values
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
    else:
        try:
            wd, m = hp.process(ppg, sample_rate=ppg_fps)
            m['bpm'] = [m['bpm']]
        except:
            return [np.nan]

    return m['bpm']


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

def interpolate_outliers(bpm, z_score_threshold=3):
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
    
    if np.all(bpm == bpm[0]):
        return bpm

    assert np.isnan(bpm).any() == False, "The BPM array should not contain NAN, use the `interpolate_nan` function before this"
    # calculates the z-score of each point to determine outliers
    z = np.abs(stats.zscore(bpm))
    x = np.arange(len(bpm))
    # Collect `bpm` without outliers
    bpm_filtered = bpm[np.where(z < z_score_threshold)[0]]
    x = x[np.where(z < z_score_threshold)[0]]
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


if __name__ == '__main__':
    gt = np.loadtxt('/home/desafio01/Documents/Codes/bio_hmd/UBFC_DATASET/DATASET_2/subject45/ground_truth.txt')
    bpm_out = []
    segment_width = 4 # (segundos)
    stride = 1 # (segundos)
    fps = 30

    # for i in range(segment_width*fps, len(gt[0, :])+segment_width*fps, stride*fps):
    #     print(i, stride*fps, i-(segment_width*fps))
    #     bpm_out.extend(get_heartpy_bpm(gt[0, i-(segment_width*fps):i], fps, segment_width, stride))

    # print(len(gt[0,:]))
    for i in range(0, len(gt[0, :])-(segment_width*fps), stride*fps):
        # print(i, i+(segment_width*fps))
        bpm_out.extend(get_heartpy_bpm(gt[0, i:i+(segment_width*fps)], fps, segment_width, stride))

    bpm_out = interpolate_nan(bpm_out)
    if stride > 1:
        # interpolation in the values to maintain the quantity of seconds
        bpm_out = interpolate_grow(bpm_out, stride=stride)
    # Interpolation of outliers values
    bpm_out = interpolate_outliers(bpm_out, 2)

    mean_fps = np.array([gt[1, i:i+fps].mean() for i in range(0, len(gt[1, :]), fps)])
    # mean_four_sec = [mean_fps[i:i+4].mean() for i in range(0, len(mean_fps), 4)]

    plt.plot(mean_fps, label='GT do UBFC')
    plt.plot(np.arange(segment_width-3, len(bpm_out)+segment_width), get_heartpy_bpm(gt[0, :], fps, segment_width, stride, True), label='segmentwise')
    plt.plot(np.arange(segment_width, len(bpm_out)+segment_width), bpm_out, label='not segmentwise')
    plt.legend()
    plt.show()