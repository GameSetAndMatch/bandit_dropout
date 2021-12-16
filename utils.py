from typing import List
import numpy
import torch
import random
import os
import pickle as pk 

def set_random_seed(seed):
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    
def save_to_pkl(history:List, file_name:str, directory_name:str = "results/history"):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    with open(f'results/history/{file_name}.pkl', 'wb') as f:
        pk.dump(history, f)