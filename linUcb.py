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
from utils import set_random_seed

from bandit_dropout import egreedy_bandit_dropout, boltzman_bandit_dropout
from architecture import architectureMNIST
from callback import activateGradient

train_size = 4000
valid_size = 2000
batch_size = 32
transformer = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                                ])

def run_experience(nombre_entrainement=20, nombre_epoch=20, exp_name = 'linUcb',nb_buckets = 16,per_batch=True, reward=None, reward_type='accuracy', seed=None):

    set_random_seed(seed)
    dataset_CIFAR10 =  datasets.CIFAR10(root='./data', train=True, download=True, transform=transformer)
    train_dataset_CIFAR10, valid_dataset_CIFAR10, test_dataset_CIFAR10 = random_split(dataset_CIFAR10,[train_size, valid_size,len(dataset_CIFAR10)-valid_size-train_size])

    train_dataloader_CIFAR10 = DataLoader(train_dataset_CIFAR10, batch_size=32, shuffle=True)
    valid_dataloader_CIFAR10 = DataLoader(valid_dataset_CIFAR10, batch_size=32, shuffle=True)

    nb_buckets = 16
    taille_subplot = int(nb_buckets**0.5) + 1
    nb_test = 5
    taill_espace_discret = 100
    result_test = np.zeros((nb_buckets, nb_test, taill_espace_discret))
    history_list = list()

    for test_indice in range(nb_test):

        dropout = linucb_bandit_dropout(nb_buckets=nb_buckets,batch_update=per_batch,dropout_max=0.8, p=0.5)
        dropout.triggered = True
        modele = architectureCIFAR10(dropout)
        pt_modele = pt.Model(modele, "sgd", "cross_entropy", batch_metrics=["accuracy"])
        history = pt_modele.fit_generator(train_dataloader_CIFAR10,valid_dataloader_CIFAR10,epochs=20,callbacks=[activateGradientlinUCB(test_dataset_CIFAR10,100,reward_type=reward_type)])
        history_list.append(history)
        for bucket in range(dropout.nb_buckets):
            f_hat_x = dropout.phi_X.dot(np.linalg.inv(dropout.V_t[bucket]).dot(dropout.B[bucket]))
            result_test[bucket,test_indice,:] = f_hat_x



    
    with open(f'history/{exp_name}.pkl', 'wb') as f:
        pk.dump(history_list, f)

    result_test 
    fig,ax = plt.subplots(taille_subplot,taille_subplot)
    fig.tight_layout(pad=1.2)
    for bucket in range(dropout.nb_buckets):        
        if ((bucket - 1) < 0 ):
            ax[bucket//taille_subplot,bucket%taille_subplot].title.set_text(f'Plus petit que {np.round(float(dropout.bucket_boundaries[bucket]),1)}')
        elif (bucket + 1 == dropout.nb_buckets ):
            ax[bucket//taille_subplot,bucket%taille_subplot].title.set_text(f'Plus grand que {np.round(float(dropout.bucket_boundaries[bucket - 1]),1)}')
        else:
            ax[bucket//taille_subplot,bucket%taille_subplot].title.set_text(f'Entre {np.round(float(dropout.bucket_boundaries[bucket-1]),1)} et {np.round(float(dropout.bucket_boundaries[bucket]),1)}')
        ax[bucket//taille_subplot,bucket%taille_subplot].set_ylabel("Précision")
        ax[bucket//taille_subplot,bucket%taille_subplot].set_xlabel("Taux de dropout")
        ax[bucket//taille_subplot,bucket%taille_subplot].fill_between(dropout.discretize_structured_input,np.mean(result_test[bucket,:,:],axis=0), np.mean(result_test[bucket,:,:],axis=0) + np.std(result_test[bucket,:,:],axis=0),alpha=0.5,color='b')
        ax[bucket//taille_subplot,bucket%taille_subplot].fill_between(dropout.discretize_structured_input,np.mean(result_test[bucket,:,:],axis=0), np.mean(result_test[bucket,:,:],axis=0) - np.std(result_test[bucket,:,:],axis=0),alpha=0.5,color='b')
        ax[bucket//taille_subplot,bucket%taille_subplot].plot(dropout.discretize_structured_input,np.mean(result_test[bucket,:,:],axis=0),label=str(bucket))
        #ax[bucket//4,bucket%4].legend()
        #ax[bucket//4,bucket%4].set_ylim(42,54)
    plt.show()
    plt.savefig(f"Results/{exp_name}.png")



if __name__ == '__main__':

    run_experience(seed=42)

        