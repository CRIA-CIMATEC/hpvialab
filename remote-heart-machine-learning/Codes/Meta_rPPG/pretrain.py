import time
import random
from data import SlideWindowDataLoader
from model import meta_rPPG
from settings import TrainOptions
import sys
sys.path.append("../EVM") 
from collections import OrderedDict


opt = TrainOptions().get_options()

model = meta_rPPG(opt, isTrain=True, continue_train=opt.continue_train) # create a model given opt.model and other options
model.setup(opt) # regular setup: load and print networks; create schedulers

dataset = SlideWindowDataLoader(opt, phase = 'pretrain') # create a dataset given opt.dataset_mode and other options

# visualizer = Visualizer(opt)

per_idx = opt.per_iter_task # per_iter_task defined in settings
dataset_size = dataset.num_tasks * (dataset.task_len[0] - (opt.win_size)) # get the number of images in the dataset.
task_len = (dataset.task_len[0] - per_idx*opt.win_size) # get the lenth of the task

print(f"per_idx : {per_idx}")
print(f"dataset.num_tasks : {dataset.num_tasks}")
print(f"dataset.task_len[0] : {dataset.task_len[0]}")
print(f"dataset_size : {dataset_size}")
print(f"task_len : {task_len}")

print("Data Size: %d ||||| Batch Size: %d ||||| initial lr: %f" %
      (dataset_size, opt.batch_size, opt.lr))
task_list = random.sample(dataset.keys, opt.batch_size) # create a random list with batch_size as number of tasks
model.dataset = dataset
data = dataset[task_list, 0]
model.set_input(data) # unpack data from dataset and apply preprocessing
model.update_prototype()

loss_key = model.loss_name[0]

train_epoch_losses = OrderedDict({name : [] for name in  model.loss_name})
val_epoch_losses = OrderedDict({name : [] for name in  model.loss_name})

for epoch in range(opt.epoch_count, opt.train_epoch + 1):
   epoch_start_time = time.time() # timer for entire epoch
   i = 0
   loss = [] # initiate loss list
   avg_loss = 0
   avg_val_loss = 0
   for k in range(int(dataset.num_tasks / opt.batch_size)):
       for data_idx in range(0, task_len, 1):
          task_list = random.sample(dataset.keys, opt.batch_size) # create a random list with batch_size as number of tasks
          model.B_net.feed_hc([model.h, model.c])
          model.progress = epoch + float(data_idx)/float(task_len)
          
          for i in range(per_idx):
            supdata = dataset[task_list, data_idx + i * opt.win_size]
            model.set_input(supdata) # unpack data from dataset and apply preprocessing
            model.pretrain_theta_and_phi() # pretrain phase
            
          loss.append(model.get_current_losses(0)[2])
   # save networks  
   model.save_networks('latest')
   model.save_networks(epoch)
   # calculate average loss
   avg_loss = sum(loss) / len(loss)
   train_epoch_losses[loss_key] = avg_loss 
   val_epoch_losses[loss_key] = avg_val_loss
   
   new_lr = model.update_learning_rate(epoch)
   # print atual Epoch situation
   print('Epoch %d/%d ||||| Time: %d sec ||||| Lr: %.7f ||||| Loss: %.3f/%.3f' %
         (epoch, opt.train_epoch, time.time() - epoch_start_time, new_lr,
          avg_loss, avg_loss))
