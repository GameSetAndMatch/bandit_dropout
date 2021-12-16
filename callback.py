import poutyne as pt
from torch.utils.data import  DataLoader, random_split 



class activateGradient(pt.Callback):

    def __init__(self, data_testeur, taille_testeur, reward_type = 'accuracy'):
        super().__init__()
        self.historique_accuracy_validation = list()
        self.historique_perte_validation = list()
        self.triggered = True
        self.data = data_testeur
        self.taille_data_test = taille_testeur
        self.last_reward = None
        self.reward_type = reward_type

    def calculate_reward(self):

            data_test, _ = random_split(self.data,[self.taille_data_test,len(self.data)-self.taille_data_test])
            loss, precision = self.model.evaluate_dataset(data_test,verbose=False,batch_size=self.taille_data_test)
            if self.reward_type == 'accuracy':
                reward = precision
                self.model.network.dropout.cumulated_rewards += reward
            elif (self.reward_type == 'accuracy_increase'):

                if self.last_reward != None:
                    reward = precision-self.last_reward
                    self.model.network.dropout.cumulated_rewards += reward
                self.last_reward = precision

            elif self.reward_type == 'loss':
                reward = loss
                self.model.network.dropout.cumulated_rewards += -reward
            
            elif (self.reward_type == 'loss_increase'):

                if self.last_reward != None:
                    reward = loss-self.last_reward
                    self.model.network.dropout.cumulated_rewards += -reward
                self.last_reward = loss

    def on_epoch_begin(self, epoch_number, logs):
        if not self.model.network.dropout.batch_update:
            self.model.network.dropout.get_dropout_rate_per_arm()

    def on_train_batch_begin(self, epoch_number, logs):
        if self.model.network.dropout.batch_update:
            self.model.network.dropout.get_dropout_rate_per_arm()

    def on_epoch_end(self, batch, logs):
          ## À chaque début d'epoch
        self.historique_accuracy_validation.append(logs["val_acc"])  
        self.historique_perte_validation.append(logs["val_loss"])

        if not self.model.network.dropout.batch_update:               
            self.calculate_reward()




    def on_train_batch_end(self, batch, logs):

        if self.model.network.dropout.batch_update:                
            self.calculate_reward()



class activateGradientlinUCB(pt.Callback):

    def __init__(self, data_testeur, taille_testeur):
        super().__init__()
        self.historique_accuracy_validation = list()
        self.historique_perte_validation = list()
        self.triggered = False
        self.data = data_testeur
        self.taille_data_test = taille_testeur
        self.last_loss = 0
        self.last_precision = None

    def on_epoch_begin(self, epoch_number, logs):
        if not self.model.network.dropout.batch_update:
            self.model.network.dropout.epoch_dropout_rate = self.model.network.dropout.choose_dropout_linucb()

    def on_epoch_end(self, batch, logs):
          ## À chaque début d'epoch
        self.historique_accuracy_validation.append(logs["val_acc"])  
        self.historique_perte_validation.append(logs["val_loss"])
        if (len(self.historique_perte_validation)>1):
            if (self.historique_perte_validation[-2] < self.historique_perte_validation[-1]):
                self.triggered = True
                self.model.network.dropout.triggered = True

        
        if (self.model.network.dropout.triggered and not self.model.network.dropout.batch_update):
            self.calculate_reward()

    def on_train_batch_end(self, batch, logs):

        if (self.model.network.dropout.triggered and self.model.network.dropout.batch_update):
            self.calculate_reward()

    def calculate_reward(self):

            data_test, _ = random_split(self.data,[self.taille_data_test,len(self.data)-self.taille_data_test])
            loss, precision = self.model.evaluate_dataset(data_test,verbose=False,batch_size=self.taille_data_test)
            if self.reward_type == 'accuracy':
                reward = precision
                self.model.network.dropout.update_bandit(reward)
            elif (self.reward_type == 'accuracy_increase'):

                if self.last_reward != None:
                    reward = precision-self.last_reward
                    self.model.network.dropout.update_bandit(reward)
                self.last_reward = precision

            elif self.reward_type == 'loss':
                reward = loss
                self.model.network.dropout.update_bandit(-reward)
            
            elif (self.reward_type == 'loss_increase'):

                if self.last_reward != None:
                    reward = loss-self.last_reward
                    self.model.network.dropout.update_bandit(-reward)
                self.last_reward = loss
            

            