import pickle

import torch

Task2 = torch.load('C:/Users/LIAN/Desktop/VSCODE/SSL-ECG-Paper-Reimplementaton/model_data/task_head_0.pt', map_location=torch.device('cpu'))

Task2['head_3.weight']= Task2['head_2.weight']
Task2['head_3.bias']= Task2['head_2.bias']

print(type(Task2))

del Task2['head_2.weight']
del Task2['head_2.bias']

print(Task2)
torch.save(Task2, 'D:/EMOTION PREDICT/SSL-ECG-Paper-Reimplementaton/model_data/TskCrrc0.pt')

TaskCrrc = torch.load('D:/EMOTION PREDICT/SSL-ECG-Paper-Reimplementaton/model_data/TskCrrc0.pt', map_location=torch.device('cpu'))
print('-'*100)
print(TaskCrrc)