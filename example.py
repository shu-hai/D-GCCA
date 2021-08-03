import numpy as np
import dgcca

n_sets = 3

Yk = [[] for _ in range(n_sets)]

for i in range(n_sets):
    Yk[i] = np.load('Y' + str(i) + '.npy')
    
Xk_hat, Ck_hat, Dk_hat, PVE_by_Ck_SetLevel_hat,  PVE_by_Ck_VariableLevel_hat  = dgcca.dGCCA(Yk, method='ED', sig_level=0.05, bootstrap_samples=10000)
                   



            
                
