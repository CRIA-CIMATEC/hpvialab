from __future__ import print_function
import numpy as np
import torch
from scipy import signal
from scipy.signal import butter

def butter_bandpass(lowcut, highcut, fs, order=5):
   '''Method to calculate butter bandpass'''
   nyq = 0.5 * fs
   low = lowcut / nyq
   high = highcut / nyq
   b, a = butter(order, [low, high], btype='band')
   return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
   '''Method to appy butter bandpass filter'''
   b, a = butter_bandpass(lowcut, highcut, fs, order=order)
   # y = lfilter(b, a, data)
   y = signal.filtfilt(b, a, data, method="pad")
   return y


def testing(opt, model, testset, data_idx, epoch):
   '''Method that will return loss values'''
   # results, true_rPPG = model.get_current_results(0)
   loss = model.get_current_losses(0)
   key = np.random.randint(testset.num_tasks)
   test_data = testset[testset.keys[key], 0]

   # model.eval() rnn can't be adapted in eval mode
   model.set_input(test_data)
   model.fewshot_test(epoch)

   # t_results, t_true_rPPG = model.get_current_results(1)
   test_loss = model.get_current_losses(1)

   model.train()

   return loss[2], test_loss
         