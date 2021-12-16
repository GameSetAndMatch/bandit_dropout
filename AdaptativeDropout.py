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
from utils import set_random_seed, save_to_pkl
import pickle as pk



from bandit_dropout import egreedy_bandit_dropout, boltzman_bandit_dropout,dynamic_linucb_bandit_dropout

from callback import activateGradient

class architectureCIFAR10(nn.Module):

    def __init__(self):

        super(architectureCIFAR10, self).__init__()

        self.conv1 = nn.Conv2d(3,32,3)
        
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32,10,3)
 
        self.flat = nn.Flatten()
 
        self.batchnorm = nn.BatchNorm1d(1690)


        self.classification = nn.Linear(1690,10)
    
        self.dropout = Standout(self.classification,0.5, 1)


    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.flat(x)
        x = self.batchnorm(x)
        previous=x
        x = self.classification(x)
        x = F.relu(x)
        x = self.dropout(previous,x)

        return x



train_size = 4000
valid_size = 2000
batch_size = 32
transformer = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                                ])



def run_experience(seed=None, exp_name = "Adaptative_Dropout"):
    set_random_seed(seed=seed)
    dataset_CIFAR10 =  datasets.CIFAR10(root='./data', train=True, download=True, transform=transformer)
    train_dataset_CIFAR10, valid_dataset_CIFAR10, test_dataset_CIFAR10 = random_split(dataset_CIFAR10,[train_size, valid_size,len(dataset_CIFAR10)-valid_size-train_size])

    train_dataloader_CIFAR10 = DataLoader(train_dataset_CIFAR10, batch_size=32, shuffle=True)
    valid_dataloader_CIFAR10 = DataLoader(valid_dataset_CIFAR10, batch_size=32, shuffle=True)

    nb_buckets = 4
    nb_test = 20
    taill_espace_discret = 100
    history_list = []

    for test_indice in range(nb_test):

        modele = architectureCIFAR10()
        pt_modele = pt.Model(modele, "sgd", "cross_entropy", batch_metrics=["accuracy"])
        history = pt_modele.fit_generator(train_dataloader_CIFAR10,valid_dataloader_CIFAR10,epochs=20)
        history_list.append(history)

    save_to_pkl(exp_name, history_list)





if __name__ == '__main__':
    run_experience(42, "adaptative_dropout")


