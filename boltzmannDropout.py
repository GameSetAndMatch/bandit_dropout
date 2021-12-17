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
from utils import set_random_seed, save_experience
from bandit_dropout import boltzmann_bandit_dropout
from architecture import architectureMNIST
from callback import activateGradient
import pickle as pk


train_size = 4000
valid_size = 2000
batch_size = 32
transformer = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                                ])


dataset_CIFAR10 =  datasets.CIFAR10(root='./data', train=True, download=True, transform=transformer)
train_dataset_CIFAR10, valid_dataset_CIFAR10, test_dataset_CIFAR10 = random_split(dataset_CIFAR10,[train_size, valid_size,len(dataset_CIFAR10)-valid_size-train_size])

train_dataloader_CIFAR10 = DataLoader(train_dataset_CIFAR10, batch_size=32, shuffle=True)
valid_dataloader_CIFAR10 = DataLoader(valid_dataset_CIFAR10, batch_size=32, shuffle=True)


def run_experience(exp_name = 'boltzmann',nb_buckets =16, nb_arms = 4, seed=None, nombre_entrainement=20, nombre_epoch = 20, per_batch=True, reward=None, c = 1, reward_type='accuracy'):

    set_random_seed(seed)
    dataset_CIFAR10 =  datasets.CIFAR10(root='./data', train=True, download=True, transform=transformer)
    train_dataset_CIFAR10, valid_dataset_CIFAR10, test_dataset_CIFAR10 = random_split(dataset_CIFAR10,[train_size, valid_size,len(dataset_CIFAR10)-valid_size-train_size])

    train_dataloader_CIFAR10 = DataLoader(train_dataset_CIFAR10, batch_size=32, shuffle=True)
    valid_dataloader_CIFAR10 = DataLoader(valid_dataset_CIFAR10, batch_size=32, shuffle=True)


    history_list = list()
    for test_indice in range(nombre_entrainement):

        dropout = boltzmann_bandit_dropout(nb_buckets, nb_arms, dropout_min=0,dropout_max=0.8, batch_update=per_batch, c=c)
        dropout.triggered = True
        modele = architectureCIFAR10(dropout)
        pt_modele = pt.Model(modele, "sgd", "cross_entropy", batch_metrics=["accuracy"])
        history = pt_modele.fit_generator(train_dataloader_CIFAR10,valid_dataloader_CIFAR10, epochs = nombre_epoch, callbacks=[activateGradientBoltzmann(test_dataset_CIFAR10,100,reward_type=reward_type)])
        history_list.append(history)

    
    save_experience(history_list, exp_name)


if __name__ == '__main__':
    run_experience(exp_name = 'boltzmann',seed=42, nombre_entrainement=2, nombre_epoch = 2, c =1, reward_type='accuracy', per_batch=False)





