import torch
import numpy as np
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
from EVM.util.visualizer import get_bpm, to_cpu, save_plot_bpm
from EVM.util import html

def run_test(opt):
	# iter_num = opt.batch_size
	opt.phase = 'test'

	model = meta_rPPG(opt, isTrain=False, continue_train=opt.continue_train)
	model.setup(opt)

	dataset = SlideWindowDataLoader(opt, phase = opt.phase)

	per_idx = opt.per_iter_task
	# dataset_size = dataset.num_tasks * (dataset.task_len - (opt.win_size))
	# task_len = (dataset.task_len - per_idx*opt.win_size)

	print(f"per_idx : {per_idx}")
	print(f"dataset.num_tasks : {dataset.num_tasks}")
	# print(f"dataset.task_len[0] : {dataset.task_len}")
	# print(f"dataset_size : {dataset_size}")
	print(f"task_len : {dataset.task_len}")

	# total_iters = 0

	# print("Data Size: {dataset_size} ||||| Batch Size: {opt.batch_size} ||||| initial lr: {opt.lr}")
		
	out_path = os.path.join(opt.results_dir, opt.name)

	if not os.path.isdir(out_path):
		os.mkdir(out_path)

	model.dataset = dataset

	# results_array = 0
	# true_rPPG_array = 0
	videos = {}

	print("Teste")
	model.eval()

	web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.load_file))  # define the website directory
	webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.load_file), model.visual_names)

	for idx_video in dataset.keys:
		# data = dataset[idx_video, 0]
		videos[idx_video] = {key : [] for key in model.visual_names} if not idx_video in videos.keys() else videos[idx_video]
		results_array = {}

		for i in range(dataset.task_len[idx_video] // opt.win_size):
			data = dataset[[idx_video], i*opt.win_size]
			model.set_input(data)
			model.evaluate()
			result = model.get_current_results(1)
			array = results_array.get('predicted', [])
			array.extend(result['predicted'])
			results_array['predicted'] = array
			# print(result)
			
		for key in results_array.keys():
			results_array[key] = torch.cat(results_array[key], axis=0)

		for key in results_array.keys():
			videos[idx_video][key] = torch.flatten(results_array[key])
			x = videos[idx_video][key]
			x = torch.reshape(x, (len(x), 1, 1))
			x = torch.unbind(x)
			videos[idx_video][key] = to_cpu(x)
			np.save(os.path.join(webpage.get_image_dir(), key, f'{idx_video}_ppg.npy'), videos[idx_video][key])
			bpm = get_bpm(videos[idx_video][key], 4, 1, opt)
			np.save(os.path.join(webpage.get_image_dir(), key, f'{idx_video}_bpm.npy'), bpm)

			save_plot_bpm(
				webpage=webpage, 
				name=idx_video, 
				y=None,
				x=bpm, 
				stride=1, 
				predicted_fs=1
			)
	
if __name__ == '__main__':
	opt = TrainOptions().get_options()
	
	opt.batch_size = 1
	opt.load_file = 'latest'
	opt.fewshots = 0
	opt.stride_window = 1
	opt.continue_train = False
	opt.do_not_split_for_test = False

	# opt.is_raw_dataset = True
	# opt.name = 'test_pretrain_2'
	# opt.feature_image_path ='/home/victorrocha/scratch/desafio2_2021/Datasets/Dataset_hp/dataset_hp.json'

	run_test(opt)