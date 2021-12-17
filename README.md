# bandit_dropout
## Introduction
Dans ce répertoire, on utilise des bandits contextuels structurés et à actions disjointes pour choisir un taux de dropout.
Les contextes sont déterminés par les valeurs d'activations de chaque cellule. Selon ce contexte, les bandits à actions disjointes
choisissent un bras à jouer qui représente un taux de dropout. Dans le cas contextuel, le bandit choisira un taux de dropout entre 0 et 1 adapté au contexte.
Les méthodes de bandit implémentés sont les suivantes : $\epsilon$-greedy, boltzmann, linUCB et une variante de linUCB adapté pour des distributions non-stationnaires.

## Détails d'implémentation

Chaque bandit que nous avons implémenté se retrouvent dans le fichier ```bandit_dropout.py```. De plus leur mise à jour est effectué à l'aide d'un callback dans le fichier
```callback.py```. Pour chaque méthode implémentée et celle comparative à notre méthode, nous avons fait un fichier dans le répertoire. Chacun de ces fichiers
possède une fonction ```run_experience```. La fonction prend en arguments les paramètres propres à chaque méthode.

## Instructions 

  - Télécharger les librairies nécessaires : ```pip install -r requirements.txt```
  - Choisir les expériences à exécuter dans le fichier ```main.py```. Exemple de ligne à ajouter pour le test de la méthode dynamique linUCB par batch:
  ``` DynamiqueLinUcb.run_experience(batch_update = True, seed=42) ```
  - Une fois le test executé, un dossier dans résultat sera créé selon le nom de l'expérience et contiendra la perte moyenne et la précision moyenne sur l'ensemble de validation.
  Si vous faites un test de bandit contextuel structuré, vous obtiendrez aussi l'estimation de la structure selon chaque contexte. Voici les images que vous devriez obtenir pour ce test:
  <center>
       <img src="https://github.com/GameSetAndMatch/bandit_dropout/blob/master/results/Acc_dynamiclinUcb_batch/Acc_dynamiclinUcb_batch_contexte.png" width="500" height="300">
       <img src="https://github.com/GameSetAndMatch/bandit_dropout/blob/master/results/Acc_dynamiclinUcb_batch/Acc_dynamiclinUcb_batch_val_acc.png" width="500" height="300">
       <img src="https://github.com/GameSetAndMatch/bandit_dropout/blob/master/results/Acc_dynamiclinUcb_batch/Acc_dynamiclinUcb_batch_loss_acc.png" width="500" height="300">
  </center>
  
  De plus, l'historique de l'entraînement sera sauvegardé en pickle.
  

