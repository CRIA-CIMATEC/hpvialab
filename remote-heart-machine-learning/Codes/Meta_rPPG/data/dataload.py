import torch
from .pre_dataload import BaselineDataset
import random
# from Visualize.visualizer import Visualizer

class SlideWindowDataLoader():
   """Wrapper class of Dataset class that performs multi-threaded data loading.
      The class is only a container of the dataset.

      There are two ways to get a data out of the Loader. 

         1) feed in a list of videos: input = dataset[[0,3,5,10], 2020]. This gets the data starting at 2020 frame from 0, 3, 5, 10th video.
         2) feed a single value of videos:  input = dataset[0, 2020]. This gets a batch of data starting at 2020 from the 0th video.
   """

   def __init__(self, opt, phase):
      """Initialize this class
      """
      # self.visualizer = Visualizer(opt, isTrain=True)
      # self.visualizer.reset()
      self.opt = opt
      self.phase = phase

      self.dataset = BaselineDataset(opt, phase)
      print("dataset [%s-%s] was created" % ('rPPGDataset', f'{self.phase}'))
      
      self.length = int(len(self.dataset))
      print(f'self.length: {self.length}')
      self.keys = self.dataset.keys
      self.num_tasks = self.dataset.num_tasks
      self.task_len = self.dataset.task_len

   def load_data(self):
      return self

   def __len__(self):
      """Return the number of data in the dataset"""
      return self.length

   def __getitem__(self, items):
      """Return a batch of data
         items -- [task_num, index of data for specified task]
      """

      inputs = []
      ppg = []

      if self.phase in ('supp', 'query', 'pretrain'):
         batch = self.opt.batch_size
      else:
         batch = self.opt.batch_size + self.opt.fewshots
      
      if not isinstance(items[0], list):
         # rand_int = random.randint(0, int(self.task_len / self.win_size)) * 60
         for i in range(batch):
            dat = self.dataset[items[0],items[1] + 60 * i]
            inputs.append(dat['input'])
            ppg.append(dat['PPG'])
      else:
         if self.phase in ('test'): # Mudança
            for idx in items[0]:
               dat = self.dataset[idx, items[1]]
               inputs.append(dat['input'])
            inputs = torch.stack(inputs)  
            return {'input': inputs , 'rPPG': ppg}    
         else:   
            for idx in items[0]:
               dat = self.dataset[idx, items[1]]
               inputs.append(dat['input'])
               ppg.append(dat['PPG'])
      inputs = torch.stack(inputs)
      ppg = torch.stack(ppg)
      
      return {'input': inputs, 'rPPG': ppg}

   def quantify(self,rppg):
      """Quantify arry of current ppg
      
      Parameters:
         rppg: ppg batch corresponding to 1 second 
      """
      quantified = torch.empty(rppg.shape[0], dtype=torch.long)
      binary = torch.ones(rppg.shape[0], dtype=torch.long)
      tmax = rppg.max()
      tmin = rppg.min()
      interval = (tmax - tmin)/39
      for i in range(len(quantified)):
         quantified[i] = ((rppg[i] - tmin)/interval).round().long()
      return quantified

   def __call__(self):
      output_list = []
      for idx in range(self.num_tasks):
         tmp = self.dataset(idx)
         tmp['rPPG'] = tmp.pop('PPG')
         output_list.append(tmp)
      return output_list
      # pdb.set_trace()

