import torch
import torch.nn as nn
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import scipy.stats
import random
import torch
from torch.autograd import Variable
from torch import nn



import scipy.stats

from scipy.stats import norm
from itertools import chain


class random_dropout(nn.Module):

    def __init__(self, p_min=0,p_max=1):
        super(random_dropout, self).__init__()
        self.p_min = p_min
        self.p_max = p_max
    
    def get_mask(self,x,p):

        if torch.cuda.is_available():
            mask = torch.Tensor(x.shape[1]).uniform_(0, 1).cuda() < p
        else:
            mask = torch.Tensor(x.shape[1]).uniform_(0, 1) < p

        return mask.int()

    
    def forward(self,x):

        p = np.random.uniform(self.p_min,self.p_max)

        mask = self.get_mask(x,p)

        return (x*mask)/(1-p)



class homemade_dropout(nn.Module):

    def __init__(self, p):
        super(homemade_dropout, self).__init__()
        self.p = p
    
    def get_mask(self,x):

        if torch.cuda.is_available():
            mask = torch.Tensor(x.shape[1]).uniform_(0, 1).cuda() < self.p
        else:
            mask = torch.Tensor(x.shape[1]).uniform_(0, 1) < self.p

        return mask.int()

    
    def forward(self,x):

        mask = self.get_mask(x)

        return x*mask

class BernoulliBandit:
    
    def __init__(self, means, seed=None):
        '''Accept an array of K >= 2 floats in [0, 1] and (optionally)
        a seed for a random number generator.'''

        self.means = means
        self.random = np.random.RandomState(seed)
        
        # for tracking regret
        self.dropout = nn.Dropoout(dropout_before_triggered)
        self.k_star = np.argmax(means)
        self.gaps = means[self.k_star] - means
        self.regret = []
    
    def get_K(self):
        '''Return the number of actions.'''
        return len(self.means)

    def play(self, k):
        '''Accept a parameter 0 <= k < K, logs the instant pseudo-regret,
        and return the realization of a Bernoulli random variable with P(X=1)
        being the mean of the given action.'''
        self.regret.append(self.gaps[k])
        samples = self.random.rand(self.get_K())
        reward = int(samples[k] < self.means[k])
        return reward
    
    def get_cumulative_regret(self):
        '''Return an array of the cumulative sum of pseudo-regret per round.'''
        return np.cumsum(self.regret)






class egreedy_bandit_dropout(nn.Module):

<<<<<<< HEAD
    def __init__(self, nb_buckets, nb_arms_per_bucket, dropout_min = 0.0, dropout_max = 0.5, epsilon = 0.20, p=0.2,epsilon_decroissant=False):
=======
    def __init__(self, nb_buckets, nb_arms_per_bucket, dropout_min = 0.0, dropout_max = 0.5, epsilon = 0.50, p=0.2,epsilon_decroissant=False,batch_update=True):
>>>>>>> 266a4090a6b2b86bb58ee360008af438bb53c1f3
        super(egreedy_bandit_dropout, self).__init__()
        self.triggered = False
        self.bucket_boundaries = torch.Tensor(scipy.stats.norm.ppf(torch.linspace(1/nb_buckets, 1-1/nb_buckets ,nb_buckets-1)))                
        self.arms = torch.linspace(dropout_min, dropout_max, nb_arms_per_bucket)
        self.epsilon = epsilon
        self.nb_arms_per_bucket = nb_arms_per_bucket
        self.nb_buckets = nb_buckets
        self.dropout_before_triggered = nn.Dropout(p)
        self.mu_hat = torch.ones(nb_buckets, nb_arms_per_bucket)
        self.cumulated_rewards = torch.zeros(nb_buckets, nb_arms_per_bucket)
        self.nb_played = torch.zeros(nb_buckets, nb_arms_per_bucket)
        self.last_played = torch.zeros(nb_buckets, nb_arms_per_bucket)
        self.p=p
        self.epsilon_decroissant = epsilon_decroissant
        self.t = 0
        self.batch_update = batch_update
        self.dropout_rate_per_arm = None
        self.update = True


    def get_mask(self, dropout_rates):

        mask = torch.lt(dropout_rates,  torch.FloatTensor(dropout_rates.shape).uniform_(0, 1))

        return mask.int()
    
    def calculate_mu_hat(self):
        self.mu_hat = self.cumulated_rewards/self.nb_played
        
    def update_metrics(self, arms_chosen_for_each_bucket):
        self.last_played = torch.zeros(self.nb_buckets, self.nb_arms_per_bucket)
        self.last_played[torch.arange(self.nb_buckets), arms_chosen_for_each_bucket] = 1
        self.nb_played += self.last_played

    def egreedy(self) -> torch.Tensor:
        """
        implementation of egreedy strategy for each bucket that correspond to a multi-armed bandit

        Returns:
            torch.Tensor(nb_buckets,): arms chosen for each bucket
        """
        self.t += 1
        self.calculate_mu_hat()
        if self.epsilon_decroissant:
            epsilon = self.epsilon * (1/np.sqrt(self.t))
        else:
            epsilon = self.epsilon
        explore = (torch.Tensor(self.mu_hat.shape[0]).uniform_(0, 1) < epsilon).int()
        arms_chosen_for_each_bucket = (1-explore) * torch.argmax(self.mu_hat, axis = 1) + explore * torch.randint(0, self.nb_arms_per_bucket, (self.mu_hat.shape[0],))
        if self.update:
            self.update_metrics(arms_chosen_for_each_bucket)
        return arms_chosen_for_each_bucket

    def get_dropout_rate_per_arm(self):
        self.dropout_rate_per_arm = self.arms[self.egreedy()]

    def get_dropout_rate_for_each_neurons(self, x):

        dropout_rate_per_arm = self.dropout_rate_per_arm
        x_bucket = torch.bucketize(x, self.bucket_boundaries)

        return dropout_rate_per_arm[x_bucket.flatten()].reshape(x_bucket.shape)
    
    def play(self, x):
        dropout_rate = self.get_dropout_rate_for_each_neurons(x)
        return torch.nan_to_num(x * self.get_mask(dropout_rate) / (1-dropout_rate))
    
    def forward(self,x):
        self.update = x.requires_grad
        if self.triggered:
            return self.play(x)  
        else:
            return self.dropout_before_triggered(x)




class linucb_bandit_dropout(nn.Module):

    def __init__(self, nb_buckets=16, dropout_min = 0.0, dropout_max = 0.5, epsilon = 0.50, p=0.2 , discretize_size=100, features_size=4,Lambda=0.1, batch_update = True,seed=None):
        super(linucb_bandit_dropout, self).__init__()
        self.triggered = False
        self.bucket_boundaries = torch.Tensor(scipy.stats.norm.ppf(torch.linspace(1/nb_buckets, 1-1/nb_buckets ,nb_buckets-1)))                
        
        self.nb_buckets = nb_buckets
        self.dropout_before_triggered = nn.Dropout(p)


        #LinUCB
        self.batch_update = batch_update
        self.upper_bound_norme_theta = 1 # À mettre en argument
        self.upper_bound_sigma = 15 # À mettre en argument
        self.random = np.random.RandomState(seed)
        self.discretize_size = discretize_size
        self.discretize_structured_input = np.linspace(dropout_min,dropout_max,discretize_size)
        self.theta_hat = self.random.uniform(0,1,(discretize_size, nb_buckets))
        self.phi_X = np.array([[x**i for i in range(features_size)] for x in self.discretize_structured_input])
        self.Lambda = Lambda
        self.L = np.amax(np.linalg.norm(self.phi_X,axis=1))
        self.V_t = np.array([np.eye(self.phi_X.shape[1]) * self.Lambda for _ in range(self.nb_buckets)])
        self.B = np.zeros((self.nb_buckets,self.phi_X.shape[1]))
        self.indice_batch = 1
        self.epoch_dropout_rate = None

    def get_mask(self, dropout_rates):
        if (type(dropout_rates) != torch.Tensor) :
            dropout_rates = torch.Tensor(dropout_rates)
        if torch.cuda.is_available():
            mask =  torch.lt(dropout_rates,  torch.FloatTensor(dropout_rates.shape).uniform_(0, 1).cuda())
        else:
            mask = torch.lt(dropout_rates,  torch.FloatTensor(dropout_rates.shape).uniform_(0, 1))

        return mask.int()
    

        
    def update_metrics(self, last_phi_t):
        if (type(last_phi_t) == torch.Tensor):
            self.last_phi_t = np.array(last_phi_t)
        else:
            self.last_phi_t = last_phi_t



    def update_bandit(self,reward):
        
        for context in range(self.nb_buckets):
            phi_t = self.phi_X[int(self.last_phi_t[context]),:]
            self.V_t[context] += np.outer(phi_t,phi_t)
            self.B[context] += phi_t*reward




    def choose_dropout_linucb(self):
        if self.triggered:
            self.indice_batch += 1
        dropout_choosen = torch.zeros(self.nb_buckets)
        last_phi_t = torch.zeros(self.nb_buckets)
        if (self.indice_batch < 4):
            last_phi_t = np.random.randint(0,self.discretize_size,self.nb_buckets)
            dropout_choosen = self.discretize_structured_input[last_phi_t]

        else:        
            delta = np.log(self.indice_batch)
            alpha = self.upper_bound_sigma * np.sqrt(self.phi_X.shape[1] * np.log(1 + (self.indice_batch*(self.L**2)/self.Lambda)/delta)) + np.sqrt(self.Lambda*self.upper_bound_norme_theta)
            for context in range(self.nb_buckets):
                V_t_inv_context = np.linalg.inv(self.V_t[context,:,:])
                B_context = self.B[context,:]
                theta_hat_context = V_t_inv_context.dot(B_context)
                f_x_hat_context = self.phi_X.dot(theta_hat_context.T)
                upper_bound_context = alpha * (self.phi_X.dot(V_t_inv_context) * self.phi_X).sum(axis=1)
                f_ucb = f_x_hat_context + upper_bound_context
                action_choosen = np.argmax(f_ucb)
                last_phi_t[context] = action_choosen
                
                dropout_choosen[context] = self.discretize_structured_input[action_choosen]

        self.update_metrics(last_phi_t)
        return torch.Tensor(dropout_choosen)


    def get_dropout_rate_for_each_neurons(self, x):
        if self.batch_update:
            dropout_rate_per_arm = self.choose_dropout_linucb()
        else:
            dropout_rate_per_arm = self.epoch_dropout_rate
        x_bucket = torch.bucketize(x, self.bucket_boundaries)

        return dropout_rate_per_arm[x_bucket.flatten()].reshape(x_bucket.shape)
    
    def play(self, x):

        dropout_rate = self.get_dropout_rate_for_each_neurons(x)
        return torch.nan_to_num(x * self.get_mask(dropout_rate) / (1-dropout_rate))
    
    def forward(self,x):
        self.update = x.requires_grad
        if self.update:
            if self.triggered:
                
                return self.play(x)  
            else:
                return self.dropout_before_triggered(x)

        else:
            return x






class boltzmann_bandit_dropout(nn.Module):

    def __init__(self, nb_buckets, nb_arms_per_bucket, dropout_min = 0.00, dropout_max = 0.50, c = 1, p=None, batch_update = True):
        super(boltzmann_bandit_dropout, self).__init__()
        self.triggered = False
        self.bucket_boundaries = torch.tensor(norm.ppf(torch.linspace(1/nb_buckets, 1-1/nb_buckets ,nb_buckets-1)))                  
        self.arms = torch.linspace(dropout_min, dropout_max, nb_arms_per_bucket)
        self.p = (dropout_max + dropout_min)/2 if p is None else p
        self.dropout_before_triggered = nn.Dropout(self.p)
        self.c = c
        self.eta = torch.full((nb_buckets,), c)
        self.nb_arms_per_bucket = nb_arms_per_bucket
        self.nb_buckets = nb_buckets
        self.mu_hat = torch.ones(nb_buckets, nb_arms_per_bucket)
        self.cumulated_rewards = torch.ones(nb_buckets, nb_arms_per_bucket)
        self.nb_played = torch.ones(nb_buckets, nb_arms_per_bucket)
        self.last_played = torch.zeros(nb_buckets, nb_arms_per_bucket)
        self.last_played[:, int(np.floor(self.nb_arms_per_bucket/2))] = 1
        self.arms_chosen_for_each_bucket = torch.full((nb_buckets,), int(np.floor(self.nb_arms_per_bucket/2)))
        self.dropout_values = [[] for _ in range(self.nb_buckets)]
        self.arms_to_update = torch.tensor([0])
        self.choose_new_arms = True
        self.batch_update

    

    def find_min_diff(self,arr):
        n = len(arr)
        arr = sorted(arr)

        diff = np.infty

        for i in range(n-1):
            if arr[i+1] - arr[i] < diff:
                diff = arr[i+1] - arr[i]

        return max(diff, 0.1)


    def get_mask(self, dropout_rates):
        new_mask = torch.lt(dropout_rates,  torch.FloatTensor(dropout_rates.shape).uniform_(0, 1))
        self.mask = new_mask.int()
        self.choose_new_arms = False
        return self.mask
    
    def calculate_mu_hat(self):
        self.mu_hat = self.cumulated_rewards/self.nb_played
        
    def update_metrics(self, arms_chosen_for_each_bucket, arms_to_update = None):
        if arms_to_update is None:
            self.last_played = torch.zeros(self.nb_buckets, self.nb_arms_per_bucket)
            self.last_played[torch.arange(self.nb_buckets), arms_chosen_for_each_bucket] = 1
            self.nb_played += self.last_played

        else:
            self.last_played[arms_to_update, ] = 0  
            self.last_played[arms_to_update, arms_chosen_for_each_bucket[arms_to_update]] = 1
            self.nb_played += self.last_played

    def softmax(self, probs):
	    e = np.exp(probs)
	    return e / e.sum()
    
    def choose_arm_from_probs(self, probs):
        return np.random.choice(list(range(self.nb_arms_per_bucket)), size=1, p = probs)[0]


    def choose_arms_per_bucket(self) -> torch.Tensor:
        """
        implementation of boltzman strategy for each bucket that correspond to a multi-armed bandit

        Returns:
            torch.Tensor(nb_buckets,): arms chosen for each bucket
        """
        
        if self.choose_new_arms:
            self.calculate_mu_hat()
            probs = np.apply_along_axis(self.softmax , 0, self.eta * np.array(self.mu_hat.T)).T
            arms_chosen_for_each_bucket =  np.apply_along_axis(self.choose_arm_from_probs, 1, probs)
            #self.update_metrics(arms_chosen_for_each_bucket = arms_chosen_for_each_bucket, arms_to_update = self.arms_to_update)
            self.update_metrics(arms_chosen_for_each_bucket = arms_chosen_for_each_bucket)
            self.arms_to_update = torch.tensor([(self.arms_to_update + 1) % (len(self.arms) + 1)])
            self.arms_chosen_for_each_bucket = torch.argmax(self.last_played, 1)
        

        return self.arms_chosen_for_each_bucket

    def get_dropout_rate_per_arm(self):
        dropout_rate_per_arm = self.arms[self.choose_arms_per_bucket()]
        for i in range(self.nb_buckets):
            self.dropout_values[i].append(dropout_rate_per_arm[i])
        return dropout_rate_per_arm

    def get_dropout_rate_for_each_neurons(self, x):
        dropout_rate_per_arm = self.get_dropout_rate_per_arm()
        x_bucket = torch.bucketize(x, self.bucket_boundaries)

        self.dropout_rate = dropout_rate_per_arm[x_bucket.flatten()].reshape(x_bucket.shape)
        return self.dropout_rate
    
    def play(self, x):
        dropout_rate = self.get_dropout_rate_for_each_neurons(x)
        self.eta = np.full((self.nb_buckets,), self.c) * np.array(np.log(self.nb_played.sum(axis = 1)))/ np.apply_along_axis(self.find_min_diff, 1, self.mu_hat)
        return_values = torch.nan_to_num(x * self.get_mask(dropout_rate) / (1-dropout_rate))
        return return_values

    
    def forward(self,x):
        self.update = x.requires_grad
        
        if self.triggered:

            return self.play(x)  
        else:
            for i in range(self.nb_buckets):
                self.dropout_values[i].append(self.p)
            return self.dropout_before_triggered(x)






class dynamic_linucb_bandit_dropout(nn.Module):

    def __init__(self, nb_buckets=16, dropout_min = 0.0, dropout_max = 0.5, epsilon = 0.50, p=0.2 , discretize_size=100, features_size=4, Lambda=0.1, gamma = 0.995, batch_update = True ,seed=None):
        super(dynamic_linucb_bandit_dropout, self).__init__()
        self.triggered = False
        self.bucket_boundaries = torch.Tensor(scipy.stats.norm.ppf(torch.linspace(1/nb_buckets, 1-1/nb_buckets ,nb_buckets-1)))                
        
        self.nb_buckets = nb_buckets
        self.dropout_before_triggered = nn.Dropout(p)


        #LinUCB
        self.batch_update = batch_update
        self.upper_bound_norme_theta = 1 # À mettre en argument
        self.upper_bound_sigma = 15 # À mettre en argument
        self.random = np.random.RandomState(seed)
        self.discretize_size = discretize_size
        self.discretize_structured_input = np.linspace(dropout_min,dropout_max,discretize_size)
        self.theta_hat = self.random.uniform(0,1,(discretize_size, nb_buckets))
        self.phi_X = np.array([[x**i for i in range(features_size)] for x in self.discretize_structured_input])
        self.Lambda = Lambda
        self.L = np.amax(np.linalg.norm(self.phi_X,axis=1))
        self.V_t = np.array([np.eye(self.phi_X.shape[1]) * self.Lambda for _ in range(self.nb_buckets)])
        self.identite_lambda = np.array([np.eye(self.phi_X.shape[1]) * self.Lambda for _ in range(self.nb_buckets)])
        self.V_tilde_t = np.array([np.eye(self.phi_X.shape[1]) * self.Lambda for _ in range(self.nb_buckets)])
        self.B = np.zeros((self.nb_buckets,self.phi_X.shape[1]))
        self.indice_batch = 1
        self.forget_factor = gamma
        self.epoch_dropout_rate = None

    def get_mask(self, dropout_rates):
        if (type(dropout_rates) != torch.Tensor) :
            dropout_rates = torch.Tensor(dropout_rates)
        if torch.cuda.is_available():
            mask =  torch.lt(dropout_rates,  torch.FloatTensor(dropout_rates.shape).uniform_(0, 1).cuda())
        else:
            mask = torch.lt(dropout_rates,  torch.FloatTensor(dropout_rates.shape).uniform_(0, 1))

        return mask.int()
    

        
    def update_metrics(self, last_phi_t):
        if (type(last_phi_t) == torch.Tensor):
            self.last_phi_t = np.array(last_phi_t)
        else:
            self.last_phi_t = last_phi_t



    def update_bandit(self,reward):
        
        for context in range(self.nb_buckets):
            phi_t = self.phi_X[int(self.last_phi_t[context]),:]
            self.V_t[context] = self.forget_factor*self.V_t[context] +  np.outer(phi_t,phi_t) + (1-self.forget_factor)*self.identite_lambda[context]
            self.V_tilde_t[context] = (self.forget_factor**2)*self.V_tilde_t[context] +  np.outer(phi_t,phi_t) + (1-self.forget_factor**2)*self.identite_lambda[context]
            self.B[context] = self.forget_factor*self.B[context] +  phi_t*reward




    def choose_dropout_linucb(self):
        if self.triggered:
            self.indice_batch += 1
        dropout_choosen = torch.zeros(self.nb_buckets)
        last_phi_t = torch.zeros(self.nb_buckets)
        if (self.indice_batch < 4):
            last_phi_t = np.random.randint(0,self.discretize_size,self.nb_buckets)
            dropout_choosen = self.discretize_structured_input[last_phi_t]

        else:        
            delta = np.log(self.indice_batch)
            alpha = self.upper_bound_sigma * np.sqrt(self.phi_X.shape[1] * np.log(1 + (self.indice_batch*(self.L**2)/self.Lambda)/delta)) + np.sqrt(self.Lambda*self.upper_bound_norme_theta)
            alpha = np.sqrt(self.Lambda*self.upper_bound_norme_theta) + self.upper_bound_sigma * np.sqrt((2*np.log(1/delta)) + self.phi_X.shape[1]*np.log(1+((self.upper_bound_norme_theta*(1-self.forget_factor**(2*(self.indice_batch-1))))/(self.Lambda*self.phi_X.shape[1]*(1-self.forget_factor**2)))) )
            for context in range(self.nb_buckets):
                V_t_inv_context = np.linalg.inv(self.V_t[context,:,:])
                v_prime = V_t_inv_context.dot(self.V_tilde_t[context,:,:]).dot(V_t_inv_context)
                B_context = self.B[context,:]
                theta_hat_context = V_t_inv_context.dot(B_context)
                f_x_hat_context = self.phi_X.dot(theta_hat_context.T)
                upper_bound_context = alpha * (self.phi_X.dot(v_prime) * self.phi_X).sum(axis=1)
                f_ucb = f_x_hat_context + upper_bound_context
                action_choosen = np.argmax(f_ucb)
                last_phi_t[context] = action_choosen
                
                dropout_choosen[context] = self.discretize_structured_input[action_choosen]

        self.update_metrics(last_phi_t)
        return torch.Tensor(dropout_choosen)


    def get_dropout_rate_for_each_neurons(self, x):
        if self.batch_update:
            dropout_rate_per_arm = self.choose_dropout_linucb()
        else:
            dropout_rate_per_arm = self.epoch_dropout_rate
        x_bucket = torch.bucketize(x, self.bucket_boundaries)

        return dropout_rate_per_arm[x_bucket.flatten()].reshape(x_bucket.shape)
    
    def play(self, x):

        dropout_rate = self.get_dropout_rate_for_each_neurons(x)
        return torch.nan_to_num(x * self.get_mask(dropout_rate) / (1-dropout_rate))
    
    def forward(self,x):
        self.update = x.requires_grad
        if self.update:
            if self.triggered:
                
                return self.play(x)  
            else:
                return self.dropout_before_triggered(x)

        else:
            return x



class Standout(nn.Module):

    def __init__(self, last_layer, alpha, beta):
        #Taken from adaptative dropout article : https://arxiv.org/pdf/1909.09146.pdf
        print("<<<<<<<<< THIS IS DEFINETLY A STANDOUT TRAINING >>>>>>>>>>>>>>>")
        super(Standout, self).__init__()
        self.pi = last_layer.weight
        self.alpha = alpha
        self.beta = beta
        self.nonlinearity = nn.Sigmoid()


    def forward(self, previous, current, p=0.5, deterministic=False):
        # Function as in page 3 of paper: Variational Dropout
        self.p = self.nonlinearity(self.alpha * previous.matmul(self.pi.t()) + self.beta)
        self.mask = sample_mask(self.p)

        # Deterministic version as in the paper
        if(deterministic or torch.mean(self.p).data.cpu().numpy()==0):
            return self.p * current
        else:
            return self.mask * current

def sample_mask(p):
    """Given a matrix of probabilities, this will sample a mask in PyTorch."""

    if torch.cuda.is_available():
        uniform = Variable(torch.Tensor(p.size()).uniform_(0, 1).cuda())
    else:
        uniform = Variable(torch.Tensor(p.size()).uniform_(0, 1))
    mask = uniform < p

    if torch.cuda.is_available():
        mask = mask.type(torch.cuda.FloatTensor)
    else:
        mask = mask.type(torch.FloatTensor)

    return mask




class random_dropout(nn.Module):

    def __init__(self, p_min=0,p_max=1):
        super(random_dropout, self).__init__()
        self.p_min = p_min
        self.p_max = p_max
    
    def get_mask(self,x,p):

        if torch.cuda.is_available():
            mask = torch.Tensor(x.shape[1]).uniform_(0, 1).cuda() < p
        else:
            mask = torch.Tensor(x.shape[1]).uniform_(0, 1) < p

        return mask.int()

    
    def forward(self,x):

        if x.requires_grad:

            p = np.random.uniform(self.p_min,self.p_max)

            mask = self.get_mask(x,p)

            return (x*mask)/(1-p)

        else:
            return x
        

