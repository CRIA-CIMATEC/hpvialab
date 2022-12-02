from os.path import join
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from src.model import EcgNetwork
import src.data as dta
import torch
from src.constants import Constants as c
import src.utils as utils
import src.datasets.dataset_utils as du

def create_model(target_size, target_id):
    does_not_matter = len(dta.AugmentationsPretextDataset.STD_AUG) + 1
    ecg_net = EcgNetwork(does_not_matter, target_size)
    model = ecg_net.emotion_head
    embedder = ecg_net.cnn
    train_on_gpu = torch.cuda.is_available()
    device = 'cuda' if train_on_gpu else 'cpu'
    state_dict_embeddings = torch.load(f'{c.model_base_path}model_embedding.pt', map_location=torch.device(device))
    embedder.load_state_dict(state_dict_embeddings)
    state_dict_model = torch.load(f'{c.model_base_path}/{target_id}.pt',
                                  map_location=torch.device(device))
    model.load_state_dict(state_dict_model)

    return embedder, model

def run_example(target_dataset: dta.DataSets=[], target_id=None):
    dataset = dta.ds_to_constructor[target_dataset](dta.DataConstants.basepath)
    
    if c.loss == 'CrossEntropyLoss':
        target_size = 4
    elif c.loss == 'MSELoss':
        target_size = 1
    else:
        raise Exception(f'{c.loss} not implemented!')
    embedder, model = create_model(target_size, target_id)
    _, _, test_idx = np.load(f'{c.model_base_path}splits_idx.npy', allow_pickle=True)

    test_sampeler = SubsetRandomSampler(test_idx)
    test_loader = DataLoader(dataset, batch_size=1,
                             sampler=test_sampeler, num_workers=3)
    data = next(iter(test_loader))
    # data = dataset[np.random.randint(0, len(dataset))]
    emb = embedder(torch.tensor(data[0]).view(1, 1, -1).float())
    res = model(emb)

    if c.loss == 'CrossEntropyLoss':
        res = torch.argmax(res, dim=1) + 1

    res = res.detach().numpy()
    print('data: ')
    print(data[0])
    print('calculated: ')
    print(round(float(res)), 'or', res)
    print('expected: ')
    print(data[1])

def run_prediction(target_dataset: dta.DataSets=[], target_id=None):
    """Method to predict a generic dataset that are '.pkl' files inside the `data_base_path` that 
    may not have a emotion ground-truth. Since could exist several generic datasets, the user can
    choose a name to the sample with the `sample_name` at the 'constants.py' file, then the structure
    is the following: f'{data_base_path}/generic/{sample_name}/*.pkl'
    
    Keyword arguments
    -----------------
    target_dataset : int or dta.DataSets
        Number that will specify the dataset structure implemented.
    
    target_id : str
        Network head weight name to use to predict.
    
    Return
    ------
        None
    """
    
    dataset = dta.ds_to_constructor[target_dataset](dta.DataConstants.basepath)
    
    if c.loss == 'CrossEntropyLoss':
        target_size = 4
    elif c.loss == 'MSELoss':
        target_size = 1
    else:
        raise Exception(f'{c.loss} not implemented!')
    
    embedder, model = create_model(target_size, target_id)

    sample_loader = DataLoader(dataset, batch_size=1, num_workers=3)

    predictions = {}

    for i_batch, (data, label, identifier) in enumerate(utils.pbar(sample_loader, leave=False)):
        for data_batch, labels_batch, identifier_batch in zip(data, label, identifier):
            emb = embedder(torch.tensor(data_batch).view(1, 1, -1).float())
            res = model(emb)

            if c.loss == 'CrossEntropyLoss':
                res = torch.argmax(res, dim=1) + 1

            res = res.detach().numpy()

            print('='*50)
            print('Batch number:', i_batch)
            print('Identifier:', identifier_batch)
            print('Input:', data_batch)
            print('Prediction:', round(float(res)), 'or', res)
            print('Ground-truth:', labels_batch)

            section = predictions.get(identifier_batch, {})
            emotion_pred = section.get('emotion_pred', [])
            emotion_pred.append(round(float(res)))

            emotion_gt = section.get('emotion_gt', [])
            emotion_gt.append(int(labels_batch.detach().numpy()))

            section['emotion_pred'] = emotion_pred
            section['emotion_gt'] = emotion_gt
            predictions[identifier_batch] = section

    du.create_path_if_needed(join(c.results_path, c.sample_name))
    with open(join(c.results_path, c.sample_name, 'predictions.npy'), 'wb') as f:
        pickle.dump(predictions, f)