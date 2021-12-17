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
    
def save_to_pkl(history:List, file_name:str, directory_name:str = "results"):
    new_directory_name = directory_name + "/" + file_name
    if not os.path.exists(new_directory_name):
        os.makedirs(new_directory_name)
    with open(f'{new_directory_name}/{file_name}.pkl', 'wb') as f:
        pk.dump(history, f)


def save_loss_acc_plot(history, file_name, directory_name:str = "results"):

    new_directory_name = directory_name + "/" +file_name
    if not os.path.exists(new_directory_name):
        os.makedirs(new_directory_name)

    result_acc = np.zeros((len(history),len(history[0]))) 
    for i,train in enumerate(history):
        for j,epoch in enumerate(train):
            result_acc[i,j] = epoch["val_acc"]

    plt.plot(range(0,len(history[0])),np.mean(result_acc,axis=0))
    plt.fill_between(range(0,len(history[0])),np.mean(result_acc,axis=0),np.mean(result_acc,axis=0)-np.std(result_acc,axis=0),alpha=0.2,color='b')
    plt.fill_between(range(0,len(history[0])),np.mean(result_acc,axis=0),np.mean(result_acc,axis=0)+np.std(result_acc,axis=0),alpha=0.2,color='b')
    plt.title(f"{file_name} validation accuracy")
    plt.xlabel("epoch")
    plt.ylabel("Précision sur la validation")
    plt.ylim(10,55)
    plt.savefig(f"{new_directory_name}/{file_name}_val_acc.png")
    plt.close()

    result_loss = np.zeros((len(history),len(history[0]))) 
    for i,train in enumerate(history):
        for j,epoch in enumerate(train):
            result_loss[i,j] = epoch["val_loss"]

    plt.plot(range(0,len(history[0])),np.mean(result_loss,axis=0))
    plt.fill_between(range(0,len(history[0])),np.mean(result_loss,axis=0),np.mean(result_loss,axis=0)-np.std(result_loss,axis=0),alpha=0.2,color='b')
    plt.fill_between(range(0,len(history[0])),np.mean(result_loss,axis=0),np.mean(result_loss,axis=0)+np.std(result_loss,axis=0),alpha=0.2,color='b')
    plt.title(f"{file_name} validation loss")
    plt.xlabel("epoch")
    plt.ylabel("Perte sur la validation")
    plt.ylim(0,4)
    plt.savefig(f"{new_directory_name}/{file_name}_loss_acc.png")

def save_experience(history, file_name, directory_name:str = "results"):
    new_directory_name = directory_name + "/" + file_name
    if not os.path.exists(new_directory_name):
        os.makedirs(new_directory_name)
    save_to_pkl(history, file_name)
    save_loss_acc_plot(history, file_name)