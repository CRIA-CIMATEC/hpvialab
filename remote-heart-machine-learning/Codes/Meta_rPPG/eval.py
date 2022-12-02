import torch
import os
try:
	from .data import SlideWindowDataLoader
	from .model import meta_rPPG
	from .settings import TrainOptions
except:
	from data import SlideWindowDataLoader
	from model import meta_rPPG
	from settings import TrainOptions

import sys
sys.path.append('..')
from EVM.util.visualizer import compute_and_save_metrics
from EVM.util import html

def run_eval(opt):
	opt.phase = 'eval'

	model = meta_rPPG(opt, isTrain=False, continue_train=opt.continue_train) # create a model given opt.model and other options
	model.setup(opt) # regular setup: load and print networks; create schedulers

	dataset = SlideWindowDataLoader(opt, phase = opt.phase) # create a dataset given opt.dataset_mode and other options

	print(f"dataset.num_tasks : {dataset.num_tasks}")
	print(f"task_len : {dataset.task_len}")
		
	out_path = os.path.join(opt.results_dir, opt.name) # path were the results will be saved
	# check if the path exist
	if not os.path.isdir(out_path): 
		os.mkdir(out_path)

	model.dataset = dataset

	videos = {}

	model.eval()

	web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.load_file))  # define the website directory
	webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.load_file), model.visual_names)

	for idx_video in dataset.keys:
		videos[idx_video] = {key : [] for key in model.visual_names} if not idx_video in videos.keys() else videos[idx_video]
		results_array = {}
		print([idx_video])
		
		for i in range(dataset.task_len[idx_video] // opt.win_size):
			data = dataset[[idx_video], i*opt.win_size]
			model.set_input(data)
			model.evaluate()
			result = model.get_current_results(1)
			for key, value in result.items():
				array = results_array.get(key, [])
				array.extend(value)
				results_array[key] = array

		for key in results_array.keys():
			results_array[key] = torch.cat(results_array[key], axis=0)

		for key in results_array.keys():
			videos[idx_video][key] = torch.flatten(results_array[key])

			x = videos[idx_video][key]
			x = torch.reshape(x, (len(x), 1, 1))
			x = torch.unbind(x)
			videos[idx_video][key] = x

	# generate graphics, images and results
	compute_and_save_metrics(webpage, videos, opt, gt = 'ground_truth', stride_window = opt.stride_window, predicted_fs = 1)
	
if __name__ == '__main__':
	opt = TrainOptions().get_options()
	
	opt.batch_size = 1 # 25
	# opt.load_file = 'latest'
	opt.fewshots = 0

	run_eval(opt)