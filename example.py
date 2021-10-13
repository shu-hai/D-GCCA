import numpy as np
import dgcca

n_sets = 3

Yk = [[] for _ in range(n_sets)] # Note that each Yk matrix should be row-mean centered!

for i in range(n_sets):
    Yk[i] = np.load('Y' + str(i) + '.npy')
 
# Select the ranks of the Xk matrices by the ED method
Xk_hat, Ck_hat, Dk_hat, PVE_by_Ck_SetLevel_hat,  PVE_by_Ck_VariableLevel_hat  = dgcca.dGCCA(Yk, method='ED')

# Or use user-specified ranks of the Xk matrices (e.g., from the scree plot) 
# Xk_hat, Ck_hat, Dk_hat, PVE_by_Ck_SetLevel_hat,  PVE_by_Ck_VariableLevel_hat  = dgcca.dGCCA(Yk, rankX=[5,5,5])

                   
