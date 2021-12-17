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
            result_acc[i,j] = epoch["val_acc"]

    plt.plot(range(0,len(history[0])),np.mean(result_acc,axis=0))
    plt.fill_between(range(0,20),np.mean(result_acc,axis=0),np.mean(result_acc,axis=0)-np.std(result_acc,axis=0),alpha=0.2,color='b')
    plt.fill_between(range(0,20),np.mean(result_acc,axis=0),np.mean(result_acc,axis=0)+np.std(result_acc,axis=0),alpha=0.2,color='b')
    plt.title(f"{exp_name} validation accuracy")
    plt.xlabel("epoch")
    plt.ylabel("Perte sur la validation")
    plt.ylim(25,55)
    plt.legend(loc="upper right")
    plt.savefig(f"results/{exp_name}_val_acc.png")
    plt.close()
    result_loss = np.zeros((len(history),len(history[0]))) 
    for i,train in enumerate(history):
        for j,epoch in enumerate(train):
            result_loss[i,j] = epoch["val_loss"]

    plt.plot(range(0,len(history[0])),np.mean(result_loss,axis=0))
    plt.fill_between(range(0,20),np.mean(result_loss,axis=0),np.mean(result_loss,axis=0)-np.std(result_loss,axis=0),alpha=0.2,color='b')
    plt.fill_between(range(0,20),np.mean(result_loss,axis=0),np.mean(result_loss,axis=0)+np.std(result_loss,axis=0),alpha=0.2,color='b')
    plt.title(f"{exp_name} validation loss")
    plt.xlabel("epoch")
    plt.ylabel("Perte sur la validation")
    plt.ylim(0,3)
    plt.legend(loc="upper right")
    plt.savefig(f"results/{exp_name}_loss_acc.png")