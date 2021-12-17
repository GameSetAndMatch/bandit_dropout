from random import seed
import linUcb
import RandomDropout
import RegularDropout
import DynamiqueLinUcb
import EgreedyDropout
import boltzmannDropout



if __name__ == '__main__':
   nombre_epoch = 12
   nombre_entrainement = 10
   seed_exp = 42





   
   #DynamiqueLinUcb.run_experience(exp_name="Acc_Increase_dynamiclinUcb_batch",  reward_type="accuracy_increase",nombre_epoch=nombre_epoch,   nombre_entrainement=nombre_entrainement,  seed=seed_exp)
   #DynamiqueLinUcb.run_experience(exp_name="Acc_dynamiclinUcb_batch",           reward_type="accuracy",         nombre_epoch=nombre_epoch,   nombre_entrainement=nombre_entrainement,  seed=seed_exp) 
   #DynamiqueLinUcb.run_experience(exp_name="loss_Increase_dynamiclinUcb_batch", reward_type="loss_increase",    nombre_epoch=nombre_epoch,   nombre_entrainement=nombre_entrainement,  seed=seed_exp)
   DynamiqueLinUcb.run_experience(exp_name="loss_dynamiclinUcb_batch",          reward_type="loss",             nombre_epoch=nombre_epoch,   nombre_entrainement=nombre_entrainement,  seed=seed_exp)

   DynamiqueLinUcb.run_experience(exp_name="Acc_Increase_dynamiclinUcb_epoch",  reward_type="accuracy_increase",nombre_epoch=nombre_epoch,   nombre_entrainement=nombre_entrainement,  seed=seed_exp, per_batch=False)
   DynamiqueLinUcb.run_experience(exp_name="Acc_dynamiclinUcb_epoch",           reward_type="accuracy",         nombre_epoch=nombre_epoch,   nombre_entrainement=nombre_entrainement,  seed=seed_exp, per_batch=False) 
   DynamiqueLinUcb.run_experience(exp_name="loss_Increase_dynamiclinUcb_epoch", reward_type="loss_increase",    nombre_epoch=nombre_epoch,   nombre_entrainement=nombre_entrainement,  seed=seed_exp, per_batch=False)
   DynamiqueLinUcb.run_experience(exp_name="loss_dynamiclinUcb_epoch",          reward_type="loss",             nombre_epoch=nombre_epoch,   nombre_entrainement=nombre_entrainement,  seed=seed_exp, per_batch=False)


   boltzmannDropout.run_experience(exp_name="Acc_Increase_boltzmann_batch",  reward_type="accuracy_increase",  nombre_epoch=nombre_epoch,    nombre_entrainement=nombre_entrainement,seed=seed_exp)
   boltzmannDropout.run_experience(exp_name="Acc_boltzmann_batch",           reward_type="accuracy",           nombre_epoch=nombre_epoch,    nombre_entrainement=nombre_entrainement,seed=seed_exp) 
   boltzmannDropout.run_experience(exp_name="loss_Increase_boltzmann_batch", reward_type="loss_increase",      nombre_epoch=nombre_epoch,    nombre_entrainement=nombre_entrainement,seed=seed_exp)
   boltzmannDropout.run_experience(exp_name="loss_boltzmann_batch",          reward_type="loss",               nombre_epoch=nombre_epoch,    nombre_entrainement=nombre_entrainement,seed=seed_exp)

   boltzmannDropout.run_experience(exp_name="Acc_Increase_boltzmann_epoch",  reward_type="accuracy_increase",  nombre_epoch=nombre_epoch,    nombre_entrainement=nombre_entrainement,seed=seed_exp, per_batch=False)
   boltzmannDropout.run_experience(exp_name="Acc_boltzmann_epoch",           reward_type="accuracy",           nombre_epoch=nombre_epoch,    nombre_entrainement=nombre_entrainement,seed=seed_exp, per_batch=False) 
   boltzmannDropout.run_experience(exp_name="loss_Increase_boltzmann_epoch", reward_type="loss_increase",      nombre_epoch=nombre_epoch,    nombre_entrainement=nombre_entrainement,seed=seed_exp, per_batch=False)
   boltzmannDropout.run_experience(exp_name="loss_boltzmann_epoch",          reward_type="loss",               nombre_epoch=nombre_epoch,    nombre_entrainement=nombre_entrainement,seed=seed_exp, per_batch=False)

