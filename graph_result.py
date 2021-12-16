import matplotlib.pyplot as plt
import numpy as np
import pickle


random_dropout = open("result/Result_20_epoch_randomdropout_008.pkl",'rb')
random_dropout = pickle.load(random_dropout)

linucb_epoch_0_08 =  open("result/Result_20_epoch_linucb_008_notriggered.pkl",'rb')
linucb_epoch_0_08 = pickle.load(linucb_epoch_0_08)

dropout05 =  open("result/Result_20_epoch_dropout_05.pkl",'rb')
dropout05 = pickle.load(dropout05)

dropout02 =  open("result/Result_20_epoch_dropout_02.pkl",'rb')
dropout02 = pickle.load(dropout02)

nb_train = 20
nb_epoch = 20

fig, ax = plt.subplots(2,2)
indice = 0
dict_method = {"Méthode linUCB avec 16 contexte":linucb_epoch_0_08, "Dropout constant p=0.2":dropout02,"Dropout constant p=0.5":dropout05,"Dropout aléatoire entre 0 et 0.8":random_dropout}
for nom_methode,methode in dict_method.items():
    
    result = np.zeros((nb_train,nb_epoch))
    for i,train in enumerate(methode):
        for j,epoch in enumerate(train):
            result[i,j] = epoch["acc"]

    #ax[indice//2,indice%2].plot(range(0,20),np.mean(result,axis=0))
    #ax[indice//2,indice%2].fill_between(range(0,20),np.mean(result,axis=0),np.mean(result,axis=0)+np.std(result,axis=0),alpha=0.2,color='b')
    #ax[indice//2,indice%2].fill_between(range(0,20),np.mean(result,axis=0),np.mean(result,axis=0)-np.std(result,axis=0),alpha=0.2,color='b')
    #ax[indice//2,indice%2].set_ylim(30,52)
    plt.plot(range(0,20),np.mean(result,axis=0),label = nom_methode)

    indice +=1
plt.xlabel("epoch")
plt.ylabel("Perte sur la validation")
plt.legend(loc="upper right")
plt.grid()
plt.show()
    
    

  
