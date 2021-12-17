from typing import List
import numpy as np
import torch
import random
import os
import pickle as pk
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (5, 4)

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


def save_loss_acc_plot(history, file_name, directory_name:str = "results",color='b'):

    new_directory_name = directory_name + "/" +file_name
    if not os.path.exists(new_directory_name):
        os.makedirs(new_directory_name)

    result_acc = np.zeros((len(history),len(history[0]))) 
    for i,train in enumerate(history):
        for j,epoch in enumerate(train):
            result_acc[i,j] = epoch["val_acc"]

    plt.plot(range(0,len(history[0])),np.mean(result_acc,axis=0))
    plt.fill_between(range(0,len(history[0])),np.mean(result_acc,axis=0),np.mean(result_acc,axis=0)-np.std(result_acc,axis=0),alpha=0.2,color=color)
    plt.fill_between(range(0,len(history[0])),np.mean(result_acc,axis=0),np.mean(result_acc,axis=0)+np.std(result_acc,axis=0),alpha=0.2,color=color)
    plt.title(f"{file_name} validation accuracy")
    plt.xlabel("epoch")
    plt.ylabel("Précision sur la validation")
    plt.ylim(10,55)
    plt.savefig(f"{new_directory_name}/{file_name}_val_acc.png")
    plt.clf()

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
    plt.clf()

def save_experience(history, file_name, directory_name:str = "results"):
    new_directory_name = directory_name + "/" + file_name
    if not os.path.exists(new_directory_name):
        os.makedirs(new_directory_name)
    save_to_pkl(history, file_name)
    save_loss_acc_plot(history, file_name)
    plt.clf()


def compare_loss_acc_plot(file_name, directory_name:str = "results", experience_name = 'Comparaison'):

    colors = ['c','r','g','b','y']
    new_directory_name = directory_name + "/" +experience_name
    if not os.path.exists(new_directory_name):
        os.makedirs(new_directory_name)

    for color,file in zip(colors,file_name):

        history =  open(directory_name+ "/" + file + "/" + file + ".pkl",'rb')
        history = pk.load(history)

        result_acc = np.zeros((len(history),len(history[0]))) 
        for i,train in enumerate(history):
            for j,epoch in enumerate(train):
                result_acc[i,j] = epoch["val_acc"]

        plt.plot(range(0,len(history[0])),np.mean(result_acc,axis=0),label=f"{file} validation accuracy")
        plt.fill_between(range(0,len(history[0])),np.mean(result_acc,axis=0),np.mean(result_acc,axis=0)-np.std(result_acc,axis=0),alpha=0.2,color=color)
        plt.fill_between(range(0,len(history[0])),np.mean(result_acc,axis=0),np.mean(result_acc,axis=0)+np.std(result_acc,axis=0),alpha=0.2,color=color)

    plt.legend(loc="upper left")
    plt.xlabel("epoch")
    plt.ylabel("Précision sur la validation")
    plt.title(f"Précision validation {experience_name}")
    plt.savefig(f"{new_directory_name}/{experience_name}_val_acc.png",dpi=400)
    plt.clf()

    for color,file in zip(colors,file_name):

        history =  open(directory_name+ "/" + file + "/" + file + ".pkl",'rb')
        history = pk.load(history)

        result_acc = np.zeros((len(history),len(history[0]))) 
        for i,train in enumerate(history):
            for j,epoch in enumerate(train):
                result_acc[i,j] = epoch["val_loss"]

        plt.plot(range(0,len(history[0])),np.mean(result_acc,axis=0),label=f"{file} validation loss")
        plt.fill_between(range(0,len(history[0])),np.mean(result_acc,axis=0),np.mean(result_acc,axis=0)-np.std(result_acc,axis=0),alpha=0.2,color=color)
        plt.fill_between(range(0,len(history[0])),np.mean(result_acc,axis=0),np.mean(result_acc,axis=0)+np.std(result_acc,axis=0),alpha=0.2,color=color)

    plt.legend(loc="upper right")
    plt.title(f"Perte validation {experience_name}")
    plt.xlabel("epoch")
    plt.ylabel("Perte sur la validation")
    plt.savefig(f"{new_directory_name}/{experience_name}_val_loss.png",dpi=400)
    plt.clf()