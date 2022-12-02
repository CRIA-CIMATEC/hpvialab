import os
import torch

try:
	from .options.test_options import TestOptions
	from .data import create_dataset
	from .models import create_model
	from .util import html
	from .util.visualizer import compute_and_save_metrics
except:
	from options.test_options import TestOptions
	from data import create_dataset
	from models import create_model
	from util import html
	from util.visualizer import compute_and_save_metrics

from datetime import datetime

def run_test(opt):
	opt.phase = 'test'
	dataset = create_dataset(opt)     # create a dataset given opt.dataset_mode and other options
	model = create_model(opt)      # create a model given opt.model and other options
	model.setup(opt)               # regular setup: load and print networks; create schedulers
	# create a website
	web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
	if opt.load_iter > 0:  # load_iter is 0 by default
		web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
	print('creating web directory', web_dir)
	webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch), model.visual_names)
	# test with eval mode. This only affects layers like batchnorm and dropout.
	start = datetime.now()
	
	effective_num = opt.max_dataset_size

	videos = {}
	idx_video = None
	img_path = []
	
	for i, data in enumerate(dataset):
		if i >= opt.max_dataset_size:  # only apply our model to opt.num_test images
			break

		model.set_input(data)  # unpack data from data loader
		idx_video = model.subject_index[0]
		videos[idx_video] = {key : () for key in model.visual_names} if not idx_video in videos.keys() else videos[idx_video]
		
		model.evaluate()

		result = model.get_current_visuals()
		
		for key in result.keys():
			x = videos[idx_video][key] # get frames from specific video
			y = torch.reshape(result[key], (len(result[key])*len(result[key][0]), 1, 1))
			y = torch.unbind(y) # remove tensor dimension
			x = x + y   
			x = torch.FloatTensor(x)
			x = torch.reshape(x, (len(x), 1, 1))
			x = torch.unbind(x) # remove tensor dimension
			videos[idx_video][key] = x

		img_path.append(model.get_image_paths())  # get image paths

		if i % 5 == 0:  # save images to an HTML file
			print('processing (%04d)-th image: %s' % (i, img_path[i]))
	
	compute_and_save_metrics(webpage, videos, opt, stride_window=opt.stride_window)

	final = datetime.now()
	time = final - start
	secs = time.total_seconds() / effective_num
	print(f"seconds: {secs}")
	with open(os.path.join(opt.results_dir, opt.name, "mean_secs_per_img.txt"), "w") as file:
		file.write(f"average seconds per prediction: {secs} seconds") 

if __name__ == '__main__':
	opt = TestOptions().parse()  # get test options

	# hard-code some parameters for test
	opt.num_threads = 0   # test code only supports num_threads = 1
	opt.batch_size = 1    # test code only supports batch_size = 1
	opt.flip = False    # no flip; comment this line if results on flipped images are needed.
	opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
	opt.do_not_split_for_test = False
	opt.stride_window = 1

	# Debug code
	opt.name = 'test_ppg_final'
	opt.feature_image_path ='/home/victorrocha/scratch/desafio2_2021/Datasets/Dataset_hp/dataset_hp.json'
	run_test(opt)