from random import seed
from utils import compare_loss_acc_plot
import linUcb
import RandomDropout
import RegularDropout
import DynamiqueLinUcb
import EgreedyDropout
import boltzmannDropout
import matplotlib.pyplot as plt
import AdaptativeDropout



if __name__ == '__main__':
    nombre_epoch = 12
    nombre_entrainement = 10
    seed_exp = 42





   
   #DynamiqueLinUcb.run_experience(exp_name="Acc_Increase_dynamiclinUcb_batch",  reward_type="accuracy_increase",nombre_epoch=nombre_epoch,   nombre_entrainement=nombre_entrainement,              seed=seed_exp)
   #DynamiqueLinUcb.run_experience(exp_name="Acc_dynamiclinUcb_batch",           reward_type="accuracy",         nombre_epoch=nombre_epoch,   nombre_entrainement=nombre_entrainement,              seed=seed_exp) 
   #DynamiqueLinUcb.run_experience(exp_name="loss_Increase_dynamiclinUcb_batch", reward_type="loss_increase",    nombre_epoch=nombre_epoch,   nombre_entrainement=nombre_entrainement,  sigma=1.5,  seed=seed_exp)
   #DynamiqueLinUcb.run_experience(exp_name="loss_dynamiclinUcb_batch",          reward_type="loss",             nombre_epoch=nombre_epoch,   nombre_entrainement=nombre_entrainement,  sigma=1.5,  seed=seed_exp)
#
   #DynamiqueLinUcb.run_experience(exp_name="Acc_Increase_dynamiclinUcb_epoch",  reward_type="accuracy_increase",nombre_epoch=nombre_epoch,   nombre_entrainement=nombre_entrainement,  seed=seed_exp, per_batch=False)
   #DynamiqueLinUcb.run_experience(exp_name="Acc_dynamiclinUcb_epoch",           reward_type="accuracy",         nombre_epoch=nombre_epoch,   nombre_entrainement=nombre_entrainement,  seed=seed_exp, per_batch=False) 
   #DynamiqueLinUcb.run_experience(exp_name="loss_Increase_dynamiclinUcb_epoch", reward_type="loss_increase",    nombre_epoch=nombre_epoch,   nombre_entrainement=nombre_entrainement,  seed=seed_exp, per_batch=False)
   #DynamiqueLinUcb.run_experience(exp_name="loss_dynamiclinUcb_epoch",          reward_type="loss",             nombre_epoch=nombre_epoch,   nombre_entrainement=nombre_entrainement,  seed=seed_exp, per_batch=False)
#
#
   #boltzmannDropout.run_experience(exp_name="Acc_Increase_boltzmann_batch",  reward_type="accuracy_increase",  nombre_epoch=nombre_epoch,    nombre_entrainement=nombre_entrainement,seed=seed_exp)
   #boltzmannDropout.run_experience(exp_name="Acc_boltzmann_batch",           reward_type="accuracy",           nombre_epoch=nombre_epoch,    nombre_entrainement=nombre_entrainement,seed=seed_exp) 
   #boltzmannDropout.run_experience(exp_name="loss_Increase_boltzmann_batch", reward_type="loss_increase",      nombre_epoch=nombre_epoch,    nombre_entrainement=nombre_entrainement,seed=seed_exp)
   #boltzmannDropout.run_experience(exp_name="loss_boltzmann_batch",          reward_type="loss",               nombre_epoch=nombre_epoch,    nombre_entrainement=nombre_entrainement,seed=seed_exp)
#
   #boltzmannDropout.run_experience(exp_name="Acc_Increase_boltzmann_epoch",  reward_type="accuracy_increase",  nombre_epoch=nombre_epoch,    nombre_entrainement=nombre_entrainement,seed=seed_exp, per_batch=False)
   #boltzmannDropout.run_experience(exp_name="Acc_boltzmann_epoch",           reward_type="accuracy",           nombre_epoch=nombre_epoch,    nombre_entrainement=nombre_entrainement,seed=seed_exp, per_batch=False) 
   #boltzmannDropout.run_experience(exp_name="loss_Increase_boltzmann_epoch", reward_type="loss_increase",      nombre_epoch=nombre_epoch,    nombre_entrainement=nombre_entrainement,seed=seed_exp, per_batch=False)
   #boltzmannDropout.run_experience(exp_name="loss_boltzmann_epoch",          reward_type="loss",               nombre_epoch=nombre_epoch,    nombre_entrainement=nombre_entrainement,seed=seed_exp, per_batch=False)

    
    
    AdaptativeDropout.run_experience(seed=seed_exp,exp_name="Adaptative_dropout",nb_test=nombre_epoch, nb_epochs=nombre_epoch)
    
    
    
    
    
    
    #linUcb.run_experience(exp_name="Acc_Increase_linUcb_batch",reward_type="accuracy_increase",nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,seed=seed_exp)
    #plt.clf()
    #linUcb.run_experience(exp_name="Acc_linUcb_batch",reward_type="accuracy",nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,seed=seed_exp)
    #plt.clf() 
    #linUcb.run_experience(exp_name="loss_Increase_linUcb_batch",reward_type="loss_increase",nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,sigma=1.5,seed=seed_exp)
    #plt.clf()
    #linUcb.run_experience(exp_name="loss_linUcb_batch",reward_type="loss",nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,sigma=1.5,seed=seed_exp)
    #plt.clf()
    #linUcb.run_experience(exp_name="Acc_Increase_linUcb_epoch",reward_type="accuracy_increase",per_batch=False,nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,seed=seed_exp)
    #plt.clf()
    #linUcb.run_experience(exp_name="Acc_linUcb_epoch",reward_type="accuracy",per_batch=False,nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,seed=seed_exp)
    #plt.clf() 
    #linUcb.run_experience(exp_name="loss_Increase_linUcb_epoch",reward_type="loss_increase",per_batch=False,nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,sigma=1.5,seed=seed_exp)
    #plt.clf()
    #linUcb.run_experience(exp_name="loss_linUcb_epoch",reward_type="loss",per_batch=False,nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,sigma=1.5,seed=seed_exp)
    #plt.clf()

    #compare_loss_acc_plot(["Acc_Increase_linUcb_batch","Acc_Increase_linUcb_epoch"],experience_name='Comparaison_batch_epoch_linUcb_inc_acc')
    #compare_loss_acc_plot(["Acc_linUcb_batch","Acc_linUcb_epoch"],experience_name='Comparaison_batch_epoch_linUcb_acc')
    #compare_loss_acc_plot(["loss_Increase_linUcb_batch","loss_Increase_linUcb_epoch"],experience_name='Comparaison_batch_epoch_linUcb_inc_loss')
    #compare_loss_acc_plot(["loss_linUcb_batch","loss_linUcb_epoch"],experience_name='Comparaison_batch_epoch_linUcb__loss')
#
    #compare_loss_acc_plot(["Acc_Increase_linUcb_epoch","Acc_linUcb_epoch","loss_Increase_linUcb_epoch","loss_linUcb_epoch"],experience_name='Comparaison_batch_epoch_linUcb_rewards')



#
    #RandomDropout.run_experience(nombre_entrainement=nombre_entrainement,nombre_epoch=nombre_epoch,exp_name="random_dropout",seed=seed_exp)
#
    #RegularDropout.run_experience(nombre_entrainement=nombre_entrainement,nombre_epoch=nombre_epoch,p=0.5,exp_name="Reg_dropout0_5",seed=seed_exp)
    #RegularDropout.run_experience(nombre_entrainement=nombre_entrainement,nombre_epoch=nombre_epoch,p=0.2,exp_name="Reg_dropout0_2",seed=seed_exp)
#
    #
    #EgreedyDropout.run_experience(exp_name="Acc_Increase_egreedy_batch",reward_type="accuracy_increase",per_batch=True, nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,seed=seed_exp)
    #EgreedyDropout.run_experience(exp_name="Acc_egreedy_batch",reward_type="accuracy",per_batch=True,nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,seed=seed_exp) 
    #EgreedyDropout.run_experience(exp_name="loss_Increase_egreedy_batch",reward_type="loss_increase",per_batch=True,nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,seed=seed_exp)
    #EgreedyDropout.run_experience(exp_name="loss_egreedy_batch",reward_type="loss",per_batch=True,nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,seed=seed_exp)
    #EgreedyDropout.run_experience(exp_name="Acc_Increase_egreedy_epoch",reward_type="accuracy_increase",per_batch=False,nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,seed=seed_exp)
    #EgreedyDropout.run_experience(exp_name="Acc_egreedy_epoch",reward_type="accuracy",per_batch=False,nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,seed=seed_exp) 
    #EgreedyDropout.run_experience(exp_name="loss_Increase_egreedy_epoch",reward_type="loss_increase",per_batch=False,nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,seed=seed_exp)
    #EgreedyDropout.run_experience(exp_name="loss_egreedy_epoch",reward_type="loss",per_batch=False,nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,seed=seed_exp)
   #compare_loss_acc_plot(["Acc_Increase_linUcb_batch","Acc_Increase_linUcb_epoch"],experience_name='Comparaison_batch_epoch_linUcb_inc_acc')
   #compare_loss_acc_plot(["Acc_linUcb_batch","Acc_linUcb_epoch"],experience_name='Comparaison_batch_epoch_linUcb_acc')
   #compare_loss_acc_plot(["loss_Increase_linUcb_batch","loss_Increase_linUcb_epoch"],experience_name='Comparaison_batch_epoch_linUcb_inc_loss')
   #compare_loss_acc_plot(["loss_linUcb_batch","loss_linUcb_epoch"],experience_name='Comparaison_batch_epoch_linUcb__loss')
#
   #compare_loss_acc_plot(["Acc_Increase_linUcb_epoch","Acc_linUcb_epoch","loss_Increase_linUcb_epoch","loss_linUcb_epoch"],experience_name='Comparaison_batch_epoch_linUcb_rewards')




   #RandomDropout.run_experience(nombre_entrainement=nombre_entrainement,nombre_epoch=nombre_epoch,exp_name="random_dropout",seed=seed_exp)

    #compare_loss_acc_plot(["Acc_Increase_egreedy_batch","Acc_Increase_egreedy_epoch"],experience_name='Comparaison_batch_epoch_egreedy_inc_acc')
    #compare_loss_acc_plot(["Acc_egreedy_batch","Acc_egreedy_epoch"],experience_name='Comparaison_batch_epoch_egreedy_acc')
    #compare_loss_acc_plot(["loss_Increase_egreedy_batch","loss_Increase_egreedy_epoch"],experience_name='Comparaison_batch_epoch_egreedy_inc_loss')
    #compare_loss_acc_plot(["loss_egreedy_batch","loss_egreedy_epoch"],experience_name='Comparaison_batch_epoch_egreedy__loss')
    #
    #compare_loss_acc_plot(["Acc_Increase_egreedy_epoch","Acc_egreedy_epoch","loss_Increase_egreedy_epoch","loss_egreedy_epoch"],experience_name='Comparaison_batch_epoch_egreedy_rewards')    
   #RegularDropout.run_experience(nombre_entrainement=nombre_entrainement,nombre_epoch=nombre_epoch,p=0.5,exp_name="Reg_dropout0_5",seed=seed_exp)
   #RegularDropout.run_experience(nombre_entrainement=nombre_entrainement,nombre_epoch=nombre_epoch,p=0.2,exp_name="Reg_dropout0_2",seed=seed_exp)
   #
   #
   #EgreedyDropout.run_experience(exp_name="Acc_Increase_egreedy_batch",reward_type="accuracy_increase",nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,seed=seed_exp)
   #EgreedyDropout.run_experience(exp_name="Acc_egreedy_batch",reward_type="accuracy",nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,seed=seed_exp) 
   #EgreedyDropout.run_experience(exp_name="loss_Increase_egreedy_batch",reward_type="loss_increase",nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,seed=seed_exp)
   #EgreedyDropout.run_experience(exp_name="loss_egreedy_batch",reward_type="loss",nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,seed=seed_exp)
   #EgreedyDropout.run_experience(exp_name="Acc_Increase_egreedy_epoch",reward_type="accuracy_increase",per_batch=False,nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,seed=seed_exp)
   #EgreedyDropout.run_experience(exp_name="Acc_egreedy_epoch",reward_type="accuracy",per_batch=False,nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,seed=seed_exp) 
   #EgreedyDropout.run_experience(exp_name="loss_Increase_egreedy_epoch",reward_type="loss_increase",per_batch=False,nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,seed=seed_exp)
   #EgreedyDropout.run_experience(exp_name="loss_egreedy_epoch",reward_type="loss",per_batch=False,nombre_epoch=nombre_epoch,nombre_entrainement=nombre_entrainement,seed=seed_exp)


    #compare_loss_acc_plot(["Acc_Increase_egreedy_batch","Acc_Increase_egreedy_epoch"],experience_name='Comparaison_batch_epoch_egreedy_inc_acc')
    #compare_loss_acc_plot(["Acc_egreedy_batch","Acc_egreedy_epoch"],experience_name='Comparaison_batch_epoch_egreedy_acc')
    #compare_loss_acc_plot(["loss_Increase_egreedy_batch","loss_Increase_egreedy_epoch"],experience_name='Comparaison_batch_epoch_egreedy_inc_loss')
    #compare_loss_acc_plot(["loss_egreedy_batch","loss_egreedy_epoch"],experience_name='Comparaison_batch_epoch_egreedy__loss') 
    #compare_loss_acc_plot(["Acc_Increase_egreedy_epoch","Acc_egreedy_epoch","loss_Increase_egreedy_epoch","loss_egreedy_epoch"],experience_name='Comparaison_batch_epoch_egreedy_rewards')    
