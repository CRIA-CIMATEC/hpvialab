import sys

import numpy as np
sys.path.append('../artificial_dataset_creator/dataset_analyzer')
from ppg_analyzer import ppg_to_bpm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import os

cmap = {
    1: 'gray', # baseline or neutral
    2: 'brown', # stress
    3: 'gold', # amusement
    4: 'blue', # meditation
}

def plot_bpm_emotion(bpm_gt, bpm_pred, gt_emotions, pred_emotions, section_pred_gt, identifier, out_path=None, show=False):
    fig, ax = plt.subplots(4, 1, gridspec_kw={'height_ratios': [5, 1, 1, 1]}, figsize=(10, 6))
    
    ax[0].set_title(f'BPMs and emotions comparison - {identifier}')
    ax[0].plot(bpm_gt, label='BPM - Ground-truth')
    ax[0].plot(bpm_pred, label='BPM - Prediction')
    ax[0].set_xlabel('Time (seconds)')
    ax[0].set_ylabel('Beats per minute (BPM)')

    ax[1].set_title('Ground-truth emotions')
    ax[2].set_title('Predict emotions (Ground-truth)')
    ax[3].set_title('Predict emotions (Meta-rPPG)')
    ax[1].set_xlim(ax[0].get_xlim())
    ax[2].set_xlim(ax[0].get_xlim())
    ax[3].set_xlim(ax[0].get_xlim())
    ax[1].set_axis_off()
    ax[2].set_axis_off()
    ax[3].set_axis_off()

    ax[0].legend()

    handles = []

    # handles is a list, so append manual patch
    handles.append(mpatches.Patch(label='Neutral', color=cmap[1]))
    handles.append(mpatches.Patch(label='Stress', color=cmap[2]))
    handles.append(mpatches.Patch(label='Amusement', color=cmap[3]))
    handles.append(mpatches.Patch(label='Meditation', color=cmap[4]))

    current_window = 0
    for gt_window, pred_window, pred_gt_window in zip(gt_emotions, pred_emotions, section_pred_gt):
        ax[1].add_patch(Rectangle((current_window, 0), 10, 1, color=cmap[gt_window], ec='black'))

        ax[2].add_patch(Rectangle((current_window, 0), 10, 1, color=cmap[pred_gt_window], ec='black'))

        ax[3].add_patch(Rectangle((current_window, 0), 10, 1, color=cmap[pred_window], ec='black'))
        current_window += 10

    fig.tight_layout()
    # plot the legend
    ax[1].legend(handles=handles, loc='upper right', bbox_to_anchor = (1.05, 1))

    if os.path.isdir(out_path):
        fig.savefig(os.path.join(out_path, identifier+'.jpg'), bbox_inches='tight')

    if show:
        plt.show()

if __name__ == '__main__':
    gt_emotions = np.load('/home/desafio01/Documents/Codes/bio_hmd/vialab-desafio2-2021/Codes/SSL-ECG-Paper-Reimplementaton/results/mahnob_gt/predictions.npy', allow_pickle=True)
    pred_gt_emotions = np.load('/home/desafio01/Documents/Codes/bio_hmd/vialab-desafio2-2021/Codes/SSL-ECG-Paper-Reimplementaton/results/mahnob_pred_gt/predictions.npy', allow_pickle=True)
    pred_emotions = np.load('/home/desafio01/Documents/Codes/bio_hmd/vialab-desafio2-2021/Codes/SSL-ECG-Paper-Reimplementaton/results/mahnob_pred/predictions.npy', allow_pickle=True)
    # pred_emotions = dict.fromkeys(gt_emotions.keys(), {"emotion_pred": [np.random.randint(1, 5) for i in range(0, 6)]})

    path_gt_ppg = '/home/desafio01/Documents/Codes/bio_hmd/vialab-desafio2-2021/Codes/Meta_rPPG/results/five_sessions_mahnob/eval_latest/images/ground_truth/{}.npy' # test_pretrain_V4_420/eval_11
    path_pred_ppg = '/home/desafio01/Documents/Codes/bio_hmd/vialab-desafio2-2021/Codes/Meta_rPPG/results/five_sessions_mahnob/eval_latest/images/predicted/{}.npy' # test_pretrain_V4_420/eval_11

    for (id_gt, section_gt) in gt_emotions.items():
        section_pred = pred_emotions[id_gt]
        section_pred_gt = pred_gt_emotions[id_gt]

        ppg_gt = np.load(path_gt_ppg.format(id_gt))
        bpm_gt = ppg_to_bpm(ppg_gt, 30, 4, 4)

        ppg_pred = np.load(path_pred_ppg.format(id_gt))
        bpm_pred = ppg_to_bpm(ppg_pred, 30, 4, 4)

        # .split('_')[0]
        plot_bpm_emotion(bpm_gt, bpm_pred, section_gt['emotion_pred'], section_pred['emotion_pred'], section_pred_gt['emotion_pred'], identifier=id_gt, out_path='/home/desafio01/Documents/Codes/bio_hmd/vialab-desafio2-2021/Codes/SSL-ECG-Paper-Reimplementaton/results/')