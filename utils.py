from typing import List
import numpy as np
import torch
import random
import os
import pickle as pk
import matplotlib.pyplot as plt

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def save_to_pkl(history:List, file_name:str, directory_name:str = "results/history"):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    with open(f'results/history/{file_name}.pkl', 'wb') as f:
        pk.dump(history, f)


def save_loss_acc_plot(history,exp_name):

    result_acc = np.zeros((len(history),len(history[0]))) 
    for i,train in enumerate(history):
        for j,epoch in enumerate(train):
            result_acc[j,i] = epoch["val_acc"]


    #ax[indice//2,indice%2].plot(range(0,20),np.mean(result,axis=0))
    #ax[indice//2,indice%2].fill_between(range(0,20),np.mean(result,axis=0),np.mean(result,axis=0)+np.std(result,axis=0),alpha=0.2,color='b')
    #ax[indice//2,indice%2].fill_between(range(0,20),np.mean(result,axis=0),np.mean(result,axis=0)-np.std(result,axis=0),alpha=0.2,color='b')
    #ax[indice//2,indice%2].set_ylim(30,52)
    plt.plot(range(0,len(history[0])),np.mean(result,axis=0),label = nom_methode)

    plt.xlabel("epoch")
    plt.ylabel("Perte sur la validation")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()