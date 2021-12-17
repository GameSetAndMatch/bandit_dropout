from typing import List
import numpy as np
import torch
import random
import os
import pickle as pk
import matplotlib.pyplot as plt
from glob import iglob
import pandas as pd
import pyperclip

plt.rcParams["figure.figsize"] = (7, 6)

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

    plt.plot(range(0,len(history[0])),np.mean(result_acc,axis=0),color=color)
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

    plt.plot(range(0,len(history[0])),np.mean(result_loss,axis=0),color=color)
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



def get_all_last_value(directory_name:str = "results"):
    file_list = [f for f in iglob(directory_name+'/**', recursive=True) if os.path.isfile(f)]

    df = pd.DataFrame(columns=["Methode", "Validation Précision", "Écart-type Validation précision", "Validation perte", "Écart-type perte","temps moyen epoch"])

    for file in file_list:
        if ".pkl" in file:
            new_row = list()
            history =  open(file,'rb')
            history = pk.load(history)
            file = file.replace('.pkl','').split("/")[-1]
            new_row.append(file)
            precision = np.array([i[-1]["val_acc"] for i in history])
            perte = np.array([i[-1]["val_loss"] for i in history])
            temps = np.array([i[-1]["time"] for i in history])
            new_row.append(np.mean(precision))
            new_row.append(np.std(precision))
            new_row.append(np.mean(perte))
            new_row.append(np.std(perte))
            new_row.append(np.mean(temps))
            a_series = pd.Series(new_row, index = df.columns)
            df = df.append(a_series, ignore_index=True)

    return df




def type_loss(x):
    if "Acc_Increase" in x:
        return "Amélioration_précision"
    elif "Acc" in x:
        return "Précision"

    elif "loss_Increase" in x:
        return "Amélioration perte"

    elif "loss" in x:
        return "Perte"

    else:
        return "Non-applicable"

def type_batch_epoch(x):

    if "batch" in x:
        return "Mise à jour par batch"
    elif "epoch" in x:
        return "Mise à jour par epoch"
    else:
        return "Non-applicable"

def type_bandit(x):

    if "lin" in x:
        return "Actions Strcturées"
    elif "egreedy" in x:
        return "Actions disjointes"
    elif "bolt" in x:
        return "Actions disjointes"
    else:
        return "Non-applicable"

if __name__ == '__main__':
    df = get_all_last_value()
    df = df.sort_values(by=['Validation Précision'], ascending=False)
    pyperclip.copy(df.to_latex())

    df['Type reward'] = df['Methode'].apply(lambda x: type_loss(x))
    df['Type mise à jour'] = df['Methode'].apply(lambda x: type_batch_epoch(x))
    df['Type bandits'] = df['Methode'].apply(lambda x: type_bandit(x))

    df_par_reward = df.groupby('Type reward').mean()
    df_par_reward = df_par_reward.sort_values(by=['Validation Précision'], ascending=False)

    df_miseajour_type = df.groupby('Type mise à jour').mean()
    df_miseajour_type = df_miseajour_type.sort_values(by=['Validation Précision'], ascending=False)
    pyperclip.copy(df_miseajour_type.to_latex())

    df_bandit_type = df.groupby('Type bandits').mean()
    df_bandit_type = df_bandit_type.sort_values(by=['Validation Précision'], ascending=False)
    pyperclip.copy(df_bandit_type.to_latex())


