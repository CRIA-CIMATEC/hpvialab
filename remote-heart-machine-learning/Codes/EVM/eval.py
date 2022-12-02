import os
try:
    from .options.eval_options import EvalOptions
    from .data import create_dataset
    from .models import create_model
    from .util.visualizer import Visualizer, save_hr_values, compute_and_save_metrics
    from .util import html
except:
    from options.eval_options import EvalOptions
    from data import create_dataset
    from models import create_model
    from util.visualizer import Visualizer, save_hr_values, compute_and_save_metrics
    from util import html

import torch

from datetime import datetime

def run_eval(opt):
    dataset = create_dataset(opt)     # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch), model.visual_names)
    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()
    start = datetime.now()
    
    effective_num = opt.max_dataset_size

    img_path = []
    hr_values = {key : [] for key in model.visual_names}
    
    videos = {}
    idx_video = None
    
    for i, data in enumerate(dataset):
        if i >= opt.max_dataset_size:  # only apply our model to opt.num_test images
            break
           
        model.set_input(data)  # unpack data from data loader
        idx_video = model.subject_index[0]
        videos[idx_video] = {key : () for key in model.visual_names} if not idx_video in videos.keys() else videos[idx_video]
        
        model.evaluate()

        result = model.get_current_visuals()
        losses = model.get_current_losses()
        
        for key in result.keys():
            x = videos[idx_video][key] # get frames from specific video
            y = torch.reshape(result[key], (len(result[key])*len(result[key][0]), 1, 1))
            y = torch.unbind(y) # remove tensor dimension
            x = x + y   
            x = torch.FloatTensor(x)
            x = torch.reshape(x, (len(x), 1, 1))
            x = torch.unbind(x) # remove tensor dimension
            videos[idx_video][key] = x

          
        img_path.append(model.get_image_paths())    # get image paths

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image: %s' % (i, img_path[i]))
   
    
    compute_and_save_metrics(webpage, videos, opt)
    final = datetime.now()
    time = final - start
    secs = time.total_seconds() / effective_num
    print(f"seconds: {secs}")
    with open(os.path.join(opt.results_dir, opt.name, "mean_secs_per_img.txt"), "w") as file:
        file.write(f"average seconds per prediction: {secs} seconds") 

if __name__ == '__main__':
    opt = EvalOptions().parse()  # get test options

    # # hard-code some parameters for test
    # opt.num_threads = 0   # test code only supports num_threads = 1
    # opt.batch_size = 1    # test code only supports batch_size = 1
    # opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    
    # opt.load_file = 'latest'

    # opt.name = 'test_pretrain_2'
    # opt.do_not_split_for_test = False
    # opt.feature_image_path = "/home/desafio01/Documents/Codes/bio_hmd/Dataset_MR_NIRP/dataset_out_pipeline/dataset_info.json"
    # opt.gpu_ids = None

    run_eval(opt)