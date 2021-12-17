import poutyne as pt
from poutyne.framework.metrics import batch_metrics
from bandit_dropout import *
from callback import *
from architecture import *
import torchvision.datasets as datasets
from torch.utils.data import  DataLoader, random_split
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import pickle as pk
from utils import set_random_seed,save_to_pkl,save_loss_acc_plot

from bandit_dropout import *
from architecture import architectureMNIST
from callback import activateGradient

train_size = 4000
valid_size = 2000
batch_size = 32
transformer = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                                ])

def run_experience(nombre_entrainement=20,nb_epoch = 20, exp_name = 'RandomDropout', seed=None):

    set_random_seed(seed)
    dataset_CIFAR10 =  datasets.CIFAR10(root='./data', train=True, download=True, transform=transformer)
    train_dataset_CIFAR10, valid_dataset_CIFAR10, test_dataset_CIFAR10 = random_split(dataset_CIFAR10,[train_size, valid_size,len(dataset_CIFAR10)-valid_size-train_size])

    train_dataloader_CIFAR10 = DataLoader(train_dataset_CIFAR10, batch_size=32, shuffle=True)
    valid_dataloader_CIFAR10 = DataLoader(valid_dataset_CIFAR10, batch_size=32, shuffle=True)


    history_list = list()
    for test_indice in range(nombre_entrainement):

        dropout = random_dropout(p_min=0,p_max=0.8)
        dropout.triggered = True
        modele = architectureCIFAR10(dropout)
        pt_modele = pt.Model(modele, "sgd", "cross_entropy", batch_metrics=["accuracy"])
        history = pt_modele.fit_generator(train_dataloader_CIFAR10,valid_dataloader_CIFAR10,epochs=nb_epoch)
        history_list.append(history)

   
    save_to_pkl(history_list,exp_name)
    save_loss_acc_plot(history_list,exp_name)




if __name__ == '__main__':

    run_experience(seed=42,nb_epoch=2,nombre_entrainement=2)

        