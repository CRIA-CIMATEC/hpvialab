import pickle
import numpy as np
import pyedflib
from simulate import ecg_simulate, ppg_simulate
from ppg_analyzer import ppg_to_bpm
import matplotlib.pyplot as plt
import os
import sys
import glob
sys.path.append('../postprocessing')
from utils import postprocessing, plot
import neurokit2 as nk
import scipy.signal
import heartpy as hp
from xml.etree import ElementTree as ET
from scipy.io import savemat

def conversion_ecg(input_folder, output_folder):
    """Function that opens the npy file, apply the postprocessing to the PPG, generates the ECG wave \
    and creates the pickle file in the output path.

    Keyword arguments
    -----------------
    input_folder : str
        Input folder where the PPGs are located.

    output_folder : str
        Output folder where the pickle files will be saved.
    """

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for predict_path in os.listdir(input_folder):
        if 'ppg' in predict_path:
            ppg_pred = np.load(os.path.join(input_folder, predict_path))

            ppg_pred, _ = postprocessing(
                ppg_pred, 
                current_rate=30,
                desired_rate=30,
                remove_rejected=True
            )

            ecg_pred = ecg_simulate(
                current_rate=30,
                pulse_signal=ppg_pred,
                desired_rate=256,
            )


            # ecg_gt = pyedflib.highlevel.read_edf(
            #     glob.glob(os.path.join(gt_path, predict_path.split("_")[1], '*.bdf'))[0], 
            #     ch_names='EXG3'
            # ) # ECG3 (left side of abdomen)

            # ecg_sample_rate = ecg_gt[1][0]['sample_rate']
            # ecg_gt = hp.enhance_ecg_peaks(ecg_gt[0][0], ecg_gt[1][0]['sample_rate'])

            bpm_from_ppg_pred = ppg_to_bpm(ppg_pred, ppg_fps=30, stride=4, segment_width=4)
            bpm_from_ecg_pred = ppg_to_bpm(ecg_pred, ppg_fps=256, stride=4, segment_width=4)

            # signals, info = nk.ecg_process(ecg_gt, sampling_rate=ecg_sample_rate)
            # ppg_gt = ppg_simulate(
            #     ecg_gt,
            #     current_rate=ecg_sample_rate,
            #     desired_rate=30,
            # )
            # bpm_gt_ppg = ppg_to_bpm(ppg_gt, ppg_fps=30, stride=4, segment_width=4)
            
            # np.save(
            #     f'/home/desafio01/Documents/Codes/bio_hmd/vialab-desafio2-2021/Codes/Meta_rPPG/results/five_sessions_mahnob/eval_latest/images/ground_truth/{predict_path}', 
            #     ppg_gt
            # )
            # np.save(
            #     f'/home/desafio01/Documents/Codes/bio_hmd/vialab-desafio2-2021/Codes/Meta_rPPG/results/five_sessions_mahnob/eval_latest/images/predicted/{predict_path}', 
            #     ppg_pred
            # )
            # bpm_gt_ecg = scipy.signal.resample(signals['ECG_Rate'].values, int(len(signals['ECG_Rate'].values) / ecg_sample_rate))

            with open(os.path.join(output_folder, f'{predict_path.split(".")[0]}.pkl'), 'wb') as f:
                pickle.dump({
                    'label': [0],
                    'identifier': predict_path.split(".")[0],
                    'ECG': ecg_pred # ecg_pred ecg_gt
                }, f)

            plt.figure(figsize=(15, 8))
            plot(
                bpm_from_ppg_pred, bpm_from_ecg_pred, # bpm_gt_ecg, bpm_gt_ppg,
                config=[
                    # {'label': 'BPM Ground-truth'}, 
                    {'label': 'BPM from PPG'}, 
                    {'label': 'BPM from simulated ECG'}, 
                    # {'label': 'BPM Ground-truth from simulated PPG'}
                ], 
                title='BPM\'s comparison before and after ECG simulation',
                x_label='Time (seconds)', y_label='Beats per minute', show=False)
            plt.savefig(os.path.join(output_folder, f'{predict_path.split(".")[0]}_bpm.jpg'))
            plt.close()

def conversion_ppg(path_ecgs) -> None:
    """Function that opens the BDF file, retrives the value of the key named 'EXG3', generates tht PPG wave, \
    retrive the emotions from the XML file and creates the MATLAB file in the output path.

    Keyword arguments
    -----------------
    path_ecgs : str
        Input folder where the ECGs folders are located.
    """

    len_path = len(os.listdir(path_ecgs))

    # Loop to convert multiple signals
    for paths in range(len_path):

        # setting signal folder to be converted
        pulse_path1 = os.listdir(path_ecgs)[paths]
        pulse_path2 = os.listdir(os.path.join(path_ecgs, pulse_path1))[0]
        pulse_path = os.path.join(path_ecgs, pulse_path1, pulse_path2)

        print('PULSE PATH', pulse_path)

        # assert os.path.exists(pulse_path), f"The `pulse_path` parameter was not found as a file: {pulse_path}"
        out_join = os.path.join(path_ecgs, pulse_path1)
        
        # creating out_path if it doesn't exist
        if os.path.exists(os.path.join(out_join, 'PPG')):
            print('out_path does exists')
            out_path = os.path.join(out_join, 'PPG')
        else:
            print('creating out_path')
            os.makedirs(os.path.join(out_join, 'PPG'))
            out_path = os.path.join(out_join, 'PPG')   

        print('OUT_PATH', out_path)
        bdf = pyedflib.highlevel.read_edf(pulse_path, ch_names='EXG3') # ECG3 (left side of abdomen)
        assert bdf[1][0]['label'] == 'EXG3', f'The channel 34 from the BDF isn\'t the third ECG: {pulse_path}'

        # generate PPG
        ppg = ppg_simulate(
            pulse_signal=bdf[0][0],
            current_rate=bdf[1][0]['sample_rate'],
            desired_rate=30
        )

        bpm = ppg_to_bpm(bdf[0][0], ppg_fps=bdf[1][0]['sample_rate'], stride=4, segment_width=4)

        emotions_path = os.path.join(os.path.dirname(pulse_path), 'session.xml')

        emotions = {
            'feltEmo': None,
            'feltArsl': None,
            'feltVlnc': None,
            'feltCtrl': None,
            'feltPred': None
        }

        session = ET.parse(emotions_path).getroot().attrib

        for emotion in emotions.keys():
            session_emotion = session.get(emotion, np.nan)
            if session_emotion is np.nan:
                print(f'Emotion {emotion} at the following path was not found: {emotions_path}')
            emotions[emotion] = session_emotion
        
        savemat(os.path.join(out_path, 'pulseOx.mat'), {
            'bpm': bpm,
            'numPulseSample': len(ppg),
            'pulseOxRecord': ppg,
            'ecg': bdf[0][0],
            **emotions
        })

if __name__ == '__main__':
    # input_folder = os.path.join('..', 'Meta_rPPG', 'results', 'test_pretrain_V4_420', 
    #                             'eval_11', 'images', 'ground_truth') # predicted
    input_folder = os.path.join('..', 'Meta_rPPG', 'results', 'five_sessions_mahnob', 
                                'eval_latest', 'images', 'predicted') # predicted
    output_folder = 'mahnob_ecg_result'
    gt_path = os.path.join('/home', 'desafio01', 'Documents', 'Codes', 'bio_hmd', 'datasets', 'raw_mahnob', 'Sessions')

    conversion_ppg(input_folder, output_folder)
    # conversion_ecg(input_folder, output_folder)