import sys
import neurokit2

sys.path.append('../artificial_dataset_creator')

from dataset_analyzer.ppg_analyzer import ppg_to_bpm
from utils import postprocessing, get_peaks_valleys, get_periods, plot
from utils import mean_absolute_percentage_error, pearson, mean_error, rmse, stand_dev
import pandas as pd
import numpy as np
import heartpy
import os

def postprocessing_test(pred_folder, gt_folder, subjects_idx, file_pattern, sample_rate=30, remove_rejected=True, 
                        use_thr=False, title_description='', plot_heatpy_ppg=False, plot_ppg=False,
                        plot_bpm=False, plot_periods=False):
    """Function that apply the postprocessing and plots the comparison of predictions before and after it. 
	The objective of this function is to analyze the results of the postprocessing with different plots.
    It needs the ground-truths to calculate the metrics. Use `postprocessing_wo_gt` to run this code without a ground-truth.

	Keyword arguments
	-----------------
	pred_folder : str
		Path of the folder that contains each .npy file that will be processed. The default predicted folder \
            from Meta-rPPG works well with this funtion. Each npy file's name will be used to identify the plots.

	gt_folder : str
		Path of the folder that contains each .npy file that will be compared as a ground-truth. The default \
            ground_truth folder from Meta-rPPG works well with this funtion. Each npy file's name will be used \
                to identify the plots.

    subjects_idx : list
		List of subject identifiers. It will be used like this:
        >>> file_pattern = 'subject{}_ppg.npy'
        >>> for subject_idx in subjects_idx:
        >>>     ppg = np.load(file_pattern.format(subject_idx))

    sample_rate : int
        Sample rate of the PPGs.

    remove_rejected : bool
        Wheter the postprocessing should remove the rejected peaks or use it.

    use_thr : bool
        Wheter the postprocessing should be applied only to the PPGs whose exceeded the threshold (6% of rejected peaks) or to all.

    title_description : str
        Common title that will appear at every plot.

    plot_heatpy_ppg : bool
        Wheter the Heartpy analysis of the PPG should be plotted or not.

    plot_ppg : bool
        Wheter the simple plot of the PPG should be rendered or not.

    plot_bpm : bool
        Wheter the plot of the BPM should be rendered or not.

    plot_periods : bool
        Wheter the plot of the PPG periods should be rendered or not.
    
	Notes
    -----
    This method saves a file named "metrics_posprocessor.csv" with the metrics calculated.

    Return
    ------
    None
    """
    metrics = pd.DataFrame(columns=['Mean Error', 'Standard Deviation', 'RMSE', 'MAPE', 'Pearson', 
                                    'BPM_pred', 'BPM_gt', 'Rejected rate', 'Peaks_pred', 'Peaks_gt'])

    for subject_idx in subjects_idx:
        ppg_pred = np.load(os.path.join(pred_folder, file_pattern.format(subject_idx)))
        ppg_pred = neurokit2.ppg_clean(ppg_pred, sampling_rate=sample_rate)
        bpm_pred = ppg_to_bpm(ppg_pred)

        ppg_gt = np.load(os.path.join(gt_folder, file_pattern.format(subject_idx)))
        ppg_gt = neurokit2.ppg_clean(ppg_gt, sampling_rate=sample_rate)
        bpm_gt = ppg_to_bpm(ppg_gt)

        wd_gt, m_gt = heartpy.process(ppg_gt, sample_rate=sample_rate)

        wd_pred, m_pred = heartpy.process(ppg_pred, sample_rate=sample_rate)
            
        rejected_rate_pred = len(wd_pred["removed_beats"]) * 100 / len(wd_pred["peaklist"])
        
        # check if the threshold should be used or not
        if use_thr and rejected_rate_pred <= 6.:
            ppg_sim = ppg_pred
            bpm_sim = bpm_pred
        else:
            # applies the postprocessing
            ppg_sim, _ = postprocessing(ppg_pred, current_rate=sample_rate, desired_rate=30, remove_rejected=remove_rejected, random_state=None)
            # keeps the postprocessing output at the same amplitude as the ground-truth
            ppg_sim = np.interp(ppg_sim, (ppg_sim.min(), ppg_sim.max()), (ppg_gt.min(), ppg_gt.max()))
            # convert it to BPM
            bpm_sim = ppg_to_bpm(ppg_sim)

        wd_sim, m_sim = heartpy.process(ppg_sim, sample_rate=sample_rate)

        # get the PPGs periods to future plot
        gt_peaks, _ = get_peaks_valleys(ppg_gt, sample_rate, remove_rejected=remove_rejected)
        gt_periods = get_periods(gt_peaks[:, 0], sample_rate)

        pred_peaks, _ = get_peaks_valleys(ppg_pred, sample_rate, remove_rejected=remove_rejected)
        pred_periods = get_periods(pred_peaks[:, 0], sample_rate)

        sim_peaks, _ = get_peaks_valleys(ppg_sim, sample_rate, remove_rejected=remove_rejected)
        sim_periods = get_periods(sim_peaks[:, 0], sample_rate)

        if plot_heatpy_ppg:
            heartpy.plotter(wd_gt, m_gt, figsize=(10, 5), title=f'Heartpy process - Ground truth - Subject {subject_idx}')
            heartpy.plotter(wd_pred, m_pred, figsize=(10, 5), title=f'Heartpy process - Prediction - Subject {subject_idx}')
            heartpy.plotter(wd_sim, m_sim, figsize=(10, 5), title=f'Heartpy process - After post-processing - Subject {subject_idx}')

        if plot_periods:
            plot(
            gt_periods, pred_periods, sim_periods,
            config=[{'label': 'Ground-truth'}, {'label': 'Prediction'}, {'label': 'Simulated'}],
            title=f'{title_description} - Periods X Time - Subject {subject_idx}', x_label='Sequence of periods', y_label='Time (seconds)'
            )
        
        if plot_ppg:
            plot(
              ppg_gt, ppg_pred, ppg_sim, 
              config=[{'label': 'Ground-truth'}, {'label': 'Prediction'}, {'label': 'Simulated'}],
              title=f'{title_description} - PPGs - Subject {subject_idx}', x_label='Time (seconds)', y_label='PPG value', 
              x_ticks=np.arange(0, len(ppg_gt)//sample_rate)
            )
        
        if plot_bpm:
            plot(
            bpm_gt, bpm_pred, bpm_sim, 
            config=[{'label': 'Ground-truth'}, {'label': 'Prediction'}, {'label': 'Simulated'}],
            title=f'{title_description} - BPMs - Subject {subject_idx}', x_label='Time (seconds)', y_label='Beats per minute'
            )

        metrics.loc[f'subject{subject_idx}_pred'] = {
            'Mean Error': round(mean_error(bpm_gt, bpm_pred), 2), 
            'Standard Deviation': round(stand_dev(bpm_gt, bpm_pred), 2), 
            'RMSE': round(rmse(bpm_gt, bpm_pred), 2), 
            'MAPE': round(mean_absolute_percentage_error(bpm_gt, bpm_pred), 2), 
            'Pearson': round(pearson(bpm_gt, bpm_pred), 2),
            'BPM_pred': round(m_pred['bpm'], 2),
            'BPM_gt': round(bpm_gt.mean(), 2),
            'Rejected rate': round(rejected_rate_pred, 2),
            'Peaks_pred': len(wd_pred['peaklist']),
            'Peaks_gt': len(wd_gt['peaklist']),
        }
        
        rejected_rate_sim = len(wd_sim["removed_beats"]) * 100 / len(wd_sim["peaklist"])

        metrics.loc[f'subject{subject_idx}_sim'] = {
            'Mean Error': round(mean_error(bpm_gt, bpm_sim), 2), 
            'Standard Deviation': round(stand_dev(bpm_gt, bpm_sim), 2), 
            'RMSE': round(rmse(bpm_gt, bpm_sim), 2), 
            'MAPE': round(mean_absolute_percentage_error(bpm_gt, bpm_sim), 2), 
            'Pearson': round(pearson(bpm_gt, bpm_sim), 2),
            'BPM_pred': round(m_sim['bpm'], 2),
            'BPM_gt': round(bpm_gt.mean(), 2),
            'Rejected rate': round(rejected_rate_sim, 2),
            'Peaks_pred': len(wd_sim['peaklist']),
            'Peaks_gt': len(wd_gt['peaklist']),
        }

    pred_columns = [column for column in list(metrics['Mean Error'].index) if 'pred' in column]

    metrics.loc[f'mean_pred'] = {
        'Mean Error': round(metrics['Mean Error'][pred_columns].mean(), 2), 
        'Standard Deviation': round(metrics['Standard Deviation'][pred_columns].mean(), 2), 
        'RMSE': round(metrics['RMSE'][pred_columns].mean(), 2), 
        'MAPE': round(metrics['MAPE'][pred_columns].mean(), 2), 
        'Pearson': round(metrics['Pearson'][pred_columns].mean(), 2),
        'BPM_pred': round(metrics['BPM_pred'][pred_columns].mean(), 2),
        'BPM_gt': round(metrics['BPM_gt'][pred_columns].mean(), 2),
        'Rejected rate': round(metrics['Rejected rate'][pred_columns].mean(), 2),
        'Peaks_pred': round(metrics['Peaks_pred'][pred_columns].mean(), 2),
        'Peaks_gt': round(metrics['Peaks_gt'][pred_columns].mean(), 2),
    }

    sim_columns = [column for column in list(metrics['Mean Error'].index) if 'sim' in column]

    metrics.loc[f'mean_sim'] = {
        'Mean Error': round(metrics['Mean Error'][sim_columns].mean(), 2), 
        'Standard Deviation': round(metrics['Standard Deviation'][sim_columns].mean(), 2), 
        'RMSE': round(metrics['RMSE'][sim_columns].mean(), 2), 
        'MAPE': round(metrics['MAPE'][sim_columns].mean(), 2), 
        'Pearson': round(metrics['Pearson'][sim_columns].abs().mean(), 2),
        'BPM_pred': round(metrics['BPM_pred'][sim_columns].mean(), 2),
        'BPM_gt': round(metrics['BPM_gt'][sim_columns].mean(), 2),
        'Rejected rate': round(metrics['Rejected rate'][sim_columns].mean(), 2),
        'Peaks_pred': round(metrics['Peaks_pred'][sim_columns].mean(), 2),
        'Peaks_gt': round(metrics['Peaks_gt'][sim_columns].mean(), 2),
    }

    print(metrics)
    metrics.to_csv('metrics_posprocessor.csv', sep=';', encoding='utf-8', decimal=',')

def postprocessing_wo_gt(pred_folder, subjects_idx, file_pattern, sample_rate=30, remove_rejected=True, 
                        use_thr=False, title_description='', plot_heatpy_ppg=False, plot_ppg=False,
                        plot_bpm=False, plot_periods=False):
    """Function that apply the postprocessing and plots the comparison of predictions before and after it. 
	The objective of this function is to analyze the results of the postprocessing with different plots.
    It does not need the ground-truths to calculate the metrics. Use `postprocessing_test` to run this code with a ground-truth.

	Keyword arguments
	-----------------
	pred_folder : str
		Path of the folder that contains each .npy file that will be processed. The default predicted folder \
            from Meta-rPPG works well with this funtion. Each npy file's name will be used to identify the plots.

    subjects_idx : list
		List of subject identifiers. It will be used like this:
        >>> file_pattern = 'pipeline_video_ppg.npy'
        >>> pred_folder = '/path/to/pred_folder_subject{}/'
        >>> for subject_idx in subjects_idx:
        >>>     ppg = np.load(os.path.join(pred_folder.format(subject_idx), file_pattern))

    sample_rate : int
        Sample rate of the PPGs.

    remove_rejected : bool
        Wheter the postprocessing should remove the rejected peaks or use it.

    use_thr : bool
        Wheter the postprocessing should be applied only to the PPGs whose exceeded the threshold (6% of rejected peaks) or to all.

    title_description : str
        Common title that will appear at every plot.

    plot_heatpy_ppg : bool
        Wheter the Heartpy analysis of the PPG should be plotted or not.

    plot_ppg : bool
        Wheter the simple plot of the PPG should be rendered or not.

    plot_bpm : bool
        Wheter the plot of the BPM should be rendered or not.

    plot_periods : bool
        Wheter the plot of the PPG periods should be rendered or not.
    
	Notes
    -----
    This method saves a file named "metrics_wo_gt_posprocessor.csv" with the metrics calculated.

    Return
    ------
    None
    """
    metrics = pd.DataFrame(columns=['BPM_pred', 'Rejected rate', 'Peaks_pred'])

    for subject_idx in subjects_idx:
        ppg_pred = np.load(os.path.join(pred_folder.format(subject_idx), file_pattern))
        ppg_pred = neurokit2.ppg_clean(ppg_pred, sampling_rate=sample_rate)
        bpm_pred = ppg_to_bpm(ppg_pred)

        wd_pred, m_pred = heartpy.process(ppg_pred, sample_rate=sample_rate)
        
        rejected_rate_pred = len(wd_pred["removed_beats"]) * 100 / len(wd_pred["peaklist"])
        
        # check if the threshold should be used or not
        if use_thr and rejected_rate_pred <= 6.:
            ppg_sim = ppg_pred
            bpm_sim = bpm_pred
        else:
            # applies the postprocessing
            ppg_sim, _ = postprocessing(ppg_pred, current_rate=sample_rate, desired_rate=30, remove_rejected=remove_rejected, random_state=None)
            # keeps the postprocessing output at the same amplitude as the prediction
            ppg_sim = np.interp(ppg_sim, (ppg_sim.min(), ppg_sim.max()), (ppg_pred.min(), ppg_pred.max()))
            # convert it to BPM
            bpm_sim = ppg_to_bpm(ppg_sim)

        wd_sim, m_sim = heartpy.process(ppg_sim, sample_rate=sample_rate)

        # get the PPGs periods to future plot
        pred_peaks, _ = get_peaks_valleys(ppg_pred, sample_rate, remove_rejected=remove_rejected)
        pred_periods = get_periods(pred_peaks[:, 0], sample_rate)

        sim_peaks, _ = get_peaks_valleys(ppg_sim, sample_rate, remove_rejected=remove_rejected)
        sim_periods = get_periods(sim_peaks[:, 0], sample_rate)

        if plot_heatpy_ppg:
            heartpy.plotter(wd_pred, m_pred, figsize=(10, 5), title=f'Heartpy process - Prediction - Subject {subject_idx}')
            heartpy.plotter(wd_sim, m_sim, figsize=(10, 5), title=f'Heartpy process - After post-processing - Subject {subject_idx}')

        if plot_periods:
            plot(
                pred_periods, sim_periods,
                config=[{'label': 'Prediction'}, {'label': 'Simulated'}],
                title=f'{title_description} - Periods X Time - Subject {subject_idx}', x_label='Sequence of periods', y_label='Time (seconds)'
            )
        
        if plot_ppg:
            plot(
                ppg_pred, ppg_sim, 
                config=[{'label': 'Prediction'}, {'label': 'Simulated'}],
                title=f'{title_description} - PPGs - Subject {subject_idx}', x_label='Time (seconds)', y_label='PPG value', 
                x_ticks=np.arange(0, len(ppg_pred)//sample_rate)
            )
        
        if plot_bpm:
            plot(
                bpm_pred, bpm_sim, 
                config=[{'label': 'Prediction'}, {'label': 'Simulated'}],
                title=f'{title_description} - BPMs - Subject {subject_idx}', x_label='Time (seconds)', y_label='Beats per minute'
            )
        
        metrics.loc[f'subject{subject_idx}_pred'] = {
            'BPM_pred': round(m_pred['bpm'], 2),
            'Rejected rate': round(rejected_rate_pred, 2),
            'Peaks_pred': len(wd_pred['peaklist']),
        }
        
        rejected_rate_sim = len(wd_sim["removed_beats"]) * 100 / len(wd_sim["peaklist"])

        metrics.loc[f'subject{subject_idx}_sim'] = {
            'BPM_pred': round(m_sim['bpm'], 2),
            'Rejected rate': round(rejected_rate_sim, 2),
            'Peaks_pred': len(wd_sim['peaklist']),
        }

    pred_columns = [column for column in list(metrics['BPM_pred'].index) if 'pred' in column]

    metrics.loc[f'mean_pred'] = {
        'BPM_pred': round(metrics['BPM_pred'][pred_columns].mean(), 2),
        'Rejected rate': round(metrics['Rejected rate'][pred_columns].mean(), 2),
        'Peaks_pred': round(metrics['Peaks_pred'][pred_columns].mean(), 2),
    }

    sim_columns = [column for column in list(metrics['BPM_pred'].index) if 'sim' in column]

    metrics.loc[f'mean_sim'] = {
        'BPM_pred': round(metrics['BPM_pred'][sim_columns].mean(), 2),
        'Rejected rate': round(metrics['Rejected rate'][sim_columns].mean(), 2),
        'Peaks_pred': round(metrics['Peaks_pred'][sim_columns].mean(), 2),
    }

    print(metrics)
    metrics.to_csv('./metrics_wo_gt_posprocessor.csv', sep=';', encoding='utf-8', decimal=',')

if __name__ == "__main__":
    base_path = '/home/desafio01/Documents/Codes/bio_hmd/datasets/Dataset_MR_NIRP/predicts_hp_90pearson/{}/test_pretrain_V4_420/test_11/images/predicted'

    subjects = [
        'Heli3_frowning_end',
        'JasmineYan_01_laughing',
        'Rob_Brianna1_frowning_start',
        'Rob_Brianna2_laughing',
        '33s_heli'
    ]

    postprocessing_wo_gt(
        base_path, 
        file_pattern='pipeline_video_ppg.npy', # f'subject{subject_idx}__ppg.npy'
        # use_thr=True,
        # remove_rejected=False,
        subjects_idx=subjects,
        title_description='Meta-rPPG HP HMD predictions',
        # plot_heatpy_ppg=True, 
        # plot_ppg=True, 
        # plot_bpm=True, 
        # plot_periods=True
    )

    # postprocessing_test(
    #     './samples/test_pretrain_UBFC_bw/predicted', 
    #     './samples/test_pretrain_UBFC_bw/ground_truth', 
    #     # [44, 45, 46, 47],
    #     # [48, 49],
    #     [44, 45, 46, 47, 48, 49],
    #     # [46],
    #     file_pattern='subject{}__ppg.npy'
    #     # use_thr=True,
    #     # remove_rejected=False,
    #     title_description='Meta-rPPG black-and-white predictions',
    #     # plot_heatpy_ppg=True, 
    #     # plot_ppg=True, 
    #     # plot_bpm=True, 
    #     # plot_periods=True
    # )

    # TODO entender o Pearson baixo do sujeito 45 já que o rejected rate dele é menor que 6.0

    pass