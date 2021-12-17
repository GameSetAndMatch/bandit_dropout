from random import seed
import linUcb
import RandomDropout
import RegularDropout
import DynamiqueLinUcb
import EgreedyDropout
import boltzmannDropout



if __name__ == '__main__':

   linUcb.run_experience(exp_name="Acc_Increase_linUcb_batch",reward_type="accuracy_increase",nombre_epoch=12,nombre_entrainement=10,seed=42)
   linUcb.run_experience(reward_type="Acc_linUcb_batch",reward_type="accuracy",nombre_epoch=12,nombre_entrainement=10,seed=42) 
   linUcb.run_experience(exp_name="loss_Increase_linUcb_batch",reward_type="loss_increase",nombre_epoch=12,nombre_entrainement=10,seed=42)
   linUcb.run_experience(reward_type="loss_linUcb_batch",reward_type="loss",nombre_epoch=12,nombre_entrainement=10,seed=42)
   linUcb.run_experience(exp_name="Acc_Increase_linUcb_epoch",reward_type="accuracy_increase",per_batch=False,nombre_epoch=12,nombre_entrainement=10,seed=42)
   linUcb.run_experience(reward_type="Acc_linUcb_epoch",reward_type="accuracy",per_batch=False,nombre_epoch=12,nombre_entrainement=10,seed=42) 
   linUcb.run_experience(exp_name="loss_Increase_linUcb_epoch",reward_type="loss_increase",per_batch=False,nombre_epoch=12,nombre_entrainement=10,seed=42)
   linUcb.run_experience(reward_type="loss_linUcb_epoch",reward_type="loss",per_batch=False,nombre_epoch=12,nombre_entrainement=10,seed=42)

   RandomDropout.run_experience(nombre_entrainement=10,nb_epoch=12,exp_name="random_dropout",seed=42)
   RegularDropout.run_experience(nombre_entrainement=10,nombre_epoch=12,p=0.5,exp_name="Reg_dropout0_5")
   RegularDropout.run_experience(nombre_entrainement=10,nombre_epoch=12,p=0.5,exp_name="Reg_dropout0_2")

