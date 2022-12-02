import pickle
import numpy as np
from simulate import ecg_simulate
from ppg_analyzer import ppg_to_bpm
import matplotlib.pyplot as plt
import os

def plot(*arrays, config=[], title='', x_label='', y_label='', x_ticks=None, show=True):
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

input_folder = os.path.join('..', 'Meta_rPPG', 'results', 'test_pretrain_V4_420', 
                            'eval_11', 'images', 'predicted')
output_folder = 'ecg_sim_result'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for predict_path in os.listdir(input_folder):
    if 'ppg' in predict_path:
        ppg = np.load(os.path.join(input_folder, predict_path))

        ecg = ecg_simulate(
            current_rate=30,
            pulse_signal=ppg,
            desired_rate=256,
        )

        bpm_from_ppg = ppg_to_bpm(ppg, ppg_fps=30, stride=4, segment_width=4)
        bpm_from_ecg2 = ppg_to_bpm(ecg, ppg_fps=256, stride=4, segment_width=4)

        with open(os.path.join(output_folder, f'{predict_path.split(".")[0]}.pkl'), 'wb') as f:
            pickle.dump({
                'label': [0],
                'identifier': predict_path.split(".")[0],
                'ECG': ecg
            }, f)

        plt.figure(figsize=(15, 8))
        plot(
            bpm_from_ppg, bpm_from_ecg2, 
            config=[{'label': 'BPM from PPG'}, {'label': 'BPM from simulated ECG'}], 
            title='BPM\'s comparison before and after ECG simulation',
            x_label='Time (seconds)', y_label='Beats per minute', show=False)
        plt.savefig(os.path.join(output_folder, f'{predict_path.split(".")[0]}_bpm.jpg'))
        plt.close()