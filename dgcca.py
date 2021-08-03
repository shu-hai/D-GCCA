# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:01:27 2017

@author: HShu
"""
from numpy.linalg import norm
import numpy as np
import scipy as sp
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
import warnings
import scikits.bootstrap as boot

'''
Note that the data are assumed to have mean zero, so you need to centerize the data first.
'''


alpha_plus = 0
alpha_minus = 0
I_Delta_ell = None


def sPOET(Y, K=None, K_max=None, K_min=None, method=None):
#assume Y is mean-zero p*n matrix, and K is the rank of its covariance matrix.
    p,n = Y.shape
        
    U, S, V_t = sp.linalg.svd(Y, full_matrices=False)
    
    S = np.diag(S)
    
    Lambda = S**2/n # whose diagonal are eigenvalues of the sample covariance matrix
    
    #### select K
    if K is None:
        if method=='GR':
            K = select_K_by_GR(Lambda.diagonal(),K_max, K_min)
        else: 
            K = select_K_by_ED(Lambda.diagonal(),K_max, K_min)
    ####
                    
    c_hat = sum( Lambda.diagonal()[K:] ) / (p - K - p*K/n)
        
    Lambda_S = np.maximum(Lambda[:K,:K]-c_hat*p/n,0)
    
    X_hat = U[:,:K] @ np.sqrt(Lambda_S*n) @ V_t[:K,:]
    
    return X_hat, Lambda_S, U[:,:K], K # U is the Gamma matrix in Fan's Annals paper

    
def dGCCA(Y_k, method=None, rankX=None, Ell=None, I_0=None, I_Delta=None, sign_alpha=None, r_star=None, sig_level=0.05, bootstrap_samples=10000):
    # Y_k is a list containing noisy data matrices Y_1,Y_2,...,Y_K
    # method: method to select ranks
    # rankX: matrix rank of each signal matrix
    # Ell: the largest index of eigenvalue>1 of covariance matrix of all f=(f_1.T,...,f_K.T).T
    # I_0: all ell<Ell and associated alpha_ell is nonzero
    # I_Delta[ell,t]: the indicator of Delta^{(ell)}_{jk}>=0, where ell is in I_0, and t is the t-th pair of (j,k) with 1<=j<k<=num_sets; 
    #                 its shape=(len(I_0), num_sets*(num_sets-1)/2)
    # sign_alpha[ell]: the sign of alpha_ell, where ell is in I_0. A list with length=len(I_0).
    # r_star[k]: rank(cov(z_k^{I_0}))
    # sig_level: significance level of testing zero correlation
    
    global alpha_plus, alpha_minus, I_Delta_ell
    
    np.random.seed(0) # seed for the random number geneator of bootstrap
    
    num_sets = len(Y_k)
    if num_sets == 2:
        raise ValueError('The number of datasets is 2. Please use dCCA function, instead.')       
    elif num_sets < 2:
        raise ValueError('The number of datasets is less than 2.') 
        
    sample_size = Y_k[0].shape[1]
    
    F_k = [[] for _ in range(num_sets)]    
    X_k = [[] for _ in range(num_sets)]      
    Lambda_X_k = [[] for _ in range(num_sets)]  
    Lambda_X_k_inv_half = [[] for _ in range(num_sets)]  
    V_X = [[] for _ in range(num_sets)]  
    C_k = [[] for _ in range(num_sets)]  
    
    if rankX == None:
        rankX = [None for _ in range(num_sets)]  
        
    if r_star == None:
        r_star = [None for _ in range(num_sets)]      
        
    
    for k in range(num_sets):
        X_k[k], Lambda_X_k[k], V_X[k], rankX[k] = sPOET(Y=Y_k[k], K=rankX[k], method=method)
        #Lambda_X_k[k] whose diagonal entries are nonzero eigenvalues of cov(x_k)
        
        Lambda_X_k_inv_half[k] = np.zeros(Lambda_X_k[k].shape)
        index_nonzero_diag_Lambda_X_k= np.nonzero(Lambda_X_k[k].diagonal()>0)
        Lambda_X_k_inv_half[k][index_nonzero_diag_Lambda_X_k,index_nonzero_diag_Lambda_X_k] = Lambda_X_k[k].diagonal()[index_nonzero_diag_Lambda_X_k]**-0.5
        F_k[k] = Lambda_X_k_inv_half[k] @ V_X[k].T @ X_k[k]
                        

             
    F_all = np.concatenate(F_k,axis=0) 
                 
    Eta, lambda_f, _ = sp.linalg.svd( F_all @ F_all.T / sample_size )
    
        
    Eta_ell_partition = np.cumsum(rankX)
    Eta_ell_partition = np.insert(Eta_ell_partition,0,0)
    
      
    # decide Ell
    if Ell is None:                        
        Ell = sum(lambda_f>1) # will be refined later        
        if Ell > 0:            
            ell = Ell
            stop = 0
            while ell >0 and stop == 0:
                ell -= 1
                w_ell = lambda_f[ell]**-0.5 * Eta[:,ell].T @ F_all                            
                for k in range(num_sets):
                    eta_kl = Eta[Eta_ell_partition[k]:Eta_ell_partition[k+1],ell]
                    norm_eta_kl = norm(eta_kl)
                    z_kl = eta_kl.T @ F_k[k] /(norm_eta_kl*(norm_eta_kl>0)+(norm_eta_kl==0))                             
                    rejection1, _ = test_zero_corr(w_ell, z_kl, tail='right', sig_level=sig_level)                
                    if rejection1:
                        rejection2, _ = test_zero_corr(Eta[:,ell].T @ F_all - eta_kl.T @ F_k[k], z_kl, tail='right', sig_level=sig_level)
                        if rejection2:
                            stop = 1
                            break
            Ell = (ell+1)*stop           
                            
        #if Ell == 0:
        #    warnings.warn('Warning: The estimated common matrices are zero matrices!')
    
    else:
        if Ell > sum(lambda_f>0):
            warnings.warn('Your specified Ell=' + str(Ell) + ' is larger than rankF=' + str(sum(lambda_f>0)) +' that we estimated.')
        elif Ell > sum(lambda_f>1): 
            warnings.warn('Your specified Ell=' + str(Ell) + ' is larger than #(eigen(cov(f)) >1)=' + str(sum(lambda_f>1)) +' that we estimated.')
                
   
    if Ell >0:        
        Z_kl = np.zeros((num_sets, Ell, sample_size)) #z_k^{(\ell)} , \ell <= Ell       
        W_ell = np.zeros((Ell, sample_size))    
        
        #Decide I_0
        if I_0 is None:    
            I_0 = []
            
            for ell in range(Ell):
                
                rejection_ell = True
                
                W_ell[ell,:] = lambda_f[ell]**-0.5 * Eta[:,ell].T @ F_all
                                    
                for k in range(num_sets):
                    eta_kl = Eta[Eta_ell_partition[k]:Eta_ell_partition[k+1],ell]
                    norm_eta_kl = norm(eta_kl)
                    if norm_eta_kl > 0:
                        Z_kl[k,ell,:] = eta_kl.T/norm_eta_kl @ F_k[k]
                        rejection_ell, _ = test_zero_corr(W_ell[ell,:], Z_kl[k,ell,:], tail='right', sig_level=sig_level)
                        if not rejection_ell:
                            break                           
                        
                        for j in range(k):
                            rejection_ell, _ = test_zero_corr(Z_kl[j,ell,:], Z_kl[k,ell,:], tail='two', sig_level=sig_level)
                            if not rejection_ell:
                                break 
                        if not rejection_ell:
                            break 
                            
                    else:
                        break
                if rejection_ell:
                    I_0.append(ell)                                    
                                        
        else:
                        
            if len(I_0) > Ell:
                raise ValueError('The cardinarlity of your specified I_0 > Ell=' + str(Ell)+ '.')
                
            if max(I_0)>=Ell or min(I_0)<0:
                raise ValueError('One element of I_0 is not an integer in [0,Ell).')

            if sum(np.diff(np.array(I_0))<=0)>0:
                raise ValueError('I_0 should be a list of strictly increasing integers.')

         
            for ell in range(Ell):
                W_ell[ell,:] = lambda_f[ell]**-0.5 * Eta[:,ell].T @ F_all
                    
                
                for k in range(num_sets):
                    eta_kl = Eta[Eta_ell_partition[k]:Eta_ell_partition[k+1],ell]
                    norm_eta_kl = norm(eta_kl)
                    if norm_eta_kl > 0:
                        Z_kl[k,ell,:] = eta_kl.T/norm_eta_kl @ F_k[k]
                                                   
        if len(I_0) == 0:
            Ell = 0 # indicates common matrices are all zero matrices.
        else:
            if I_Delta is not None and I_Delta.shape!=(len(I_0),int(num_sets*(num_sets-1)/2)):
                ValueError('I_Delta is not well-defined. Check its dimension and type.')

            # Compute alpha_ell
            alpha_ell = np.zeros(Ell)  
            
            alpha_jk = np.zeros(int(num_sets*(num_sets-1)/2+2)) #add inf and -inf at the end
            alpha_jk[-1]=np.inf
            alpha_jk[-2]=-np.inf
            
            for ell in I_0:
                I_Delta_ell = np.ones(int(num_sets*(num_sets-1)/2))
                

                t=-1
                               
                for j in range(num_sets-1):
                    
                    corr_wl_zjl = W_ell[ell,:].T @ Z_kl[j,ell,:]/ sample_size
                    
                    for k in range(j+1,num_sets):                        
                        t = t+1
                        
                        corr_wl_zkl = W_ell[ell,:].T @ Z_kl[k,ell,:]/ sample_size
                        
                        corr_zjl_zkl = Z_kl[j,ell,:].T @ Z_kl[k,ell,:]/ sample_size
                        
                        Delta_jk = (corr_wl_zjl + corr_wl_zkl)**2 - 4*corr_zjl_zkl               
                        
                        if I_Delta is None:                                                            
                            if Delta_jk < 0:
                                Z_j0k = Z_kl[j,ell,:] - 0.5*(corr_wl_zjl + corr_wl_zkl)*W_ell[ell,:] # z_{j,k}^{(\ell)}
                                Z_k0j = Z_kl[k,ell,:] - 0.5*(corr_wl_zjl + corr_wl_zkl)*W_ell[ell,:] # z_{j,k}^{(\ell)}
                                rejection, _ = test_zero_corr(Z_j0k, Z_k0j, tail='two', sig_level=sig_level) 
                                if rejection:
                                    I_Delta_ell[t] = 0
                                    alpha_jk[t] = np.inf # indicate no alpha_jk here
                                else:
                                    alpha_jk[t] = 0.5 * (corr_wl_zjl + corr_wl_zkl) # revise Delta_jk=0
                            else:
                                alpha_jk[t] = 0.5 * (corr_wl_zjl + corr_wl_zkl - Delta_jk**0.5) 
                        else:
                            if I_Delta[I_0.index(ell),t]==0:
                                I_Delta_ell[t] = 0
                                alpha_jk[t] = np.inf # indicate no alpha_jk here
                            elif I_Delta[I_0.index(ell),t]==1:
                                Delta_jk = max(Delta_jk,0)
                                alpha_jk[t] = 0.5 * (corr_wl_zjl + corr_wl_zkl - Delta_jk**0.5) 
                            else:
                                ValueError('The entries of I_Delta must be 1 or 0.')
                
 
                
                if sign_alpha is None:
                    alpha_plus = min(alpha_jk[alpha_jk>0])
                    alpha_minus = max(alpha_jk[alpha_jk<0])
                    if alpha_plus == np.inf:
                        alpha_ell[ell] = alpha_minus
                    elif alpha_minus == -np.inf:
                        alpha_ell[ell] = alpha_plus
                    else:
                        #use BCa booststrap interval to test |alpha_plus|-|alpha_minus|==0                       
                        interval = boot.ci(data=np.concatenate((W_ell[ell:(ell+1),:].T,  Z_kl[:,ell,:].T),axis=1), 
                                           statfunction=decide_alpha_sign, alpha=sig_level, 
                                           n_samples=bootstrap_samples, method='bca')
                        
                        if (0<interval[0] or 0>interval[1]) and alpha_plus<abs(alpha_minus):
                            alpha_ell[ell] = alpha_plus
                        else:
                            alpha_ell[ell] = alpha_minus
                            
                        
                else:
                    if sign_alpha[I_0.index(ell)]>0:
                        alpha_ell[ell] = min(alpha_jk[alpha_jk>0])
                        if alpha_ell[ell] == np.inf:
                            ValueError('There does not exist alpha_ell['+str(ell)+']. Something is wrong with I_0 and sign_alpha['+str(ell)+'].')
                    elif sign_alpha[I_0.index(ell)]<0:
                        alpha_ell[ell] = max(alpha_jk[alpha_jk<0])
                        if alpha_ell[ell] == -np.inf:
                            ValueError('There does not exist alpha_ell['+str(ell)+']. Something is wrong with I_0 and sign_alpha['+str(ell)+'].')
                    else:
                        ValueError('sign_alpha['+str(ell)+'] can not be 0.')
                        
       
                
            if  sum(abs(alpha_ell)>0) == 0:
                Ell = 0                                
            else:
                for k in range(num_sets): 
                    H_k = Eta[Eta_ell_partition[k]:Eta_ell_partition[k+1],I_0].T
                    H_k = normalize(H_k,'l2',axis=1,copy=True)# normalize each row to unit L-2 norm
                    
                    cov_tilde_z_k = H_k @ H_k.T
                    
                    if r_star[k] == None:                      
                        r_star[k] = CFTtest_for_r_k_star(H_k @ F_k[k], cov_tilde_z_k, alpha=sig_level, BootstrapSize=bootstrap_samples)
                                            
                    r_k_check = min(r_star[k], np.linalg.matrix_rank(cov_tilde_z_k))
                    lambda_cov_tilde_z_k, V_cov_tilde_z_k = sp.linalg.eigh( cov_tilde_z_k, eigvals=(cov_tilde_z_k.shape[0]-r_k_check, cov_tilde_z_k.shape[0]-1) )
                    cov_hat_z_k_MPinv = V_cov_tilde_z_k @ np.diag(lambda_cov_tilde_z_k**-1) @ V_cov_tilde_z_k.T  #pseudo-inverese     
                    
                    N_mat = Eta[:,I_0].T
                    
                    A_mat = np.diag( [ alpha_ell[ell]*(lambda_f[ell]**-0.5) for ell in I_0 ] )
                    
                    C_k[k] = V_X[k] @ Lambda_X_k[k]**0.5 @ H_k.T @ cov_hat_z_k_MPinv @ A_mat @ N_mat @ F_all
                 
                
                PVE_by_Ck_SetLevel = [norm(C_k[k],'fro')**2/(norm(X_k[k],'fro')**2) for k in range(num_sets)] # X_k's variance explained by C_k at the dataset level
                PVE_by_Ck_VariableLevel = [np.sum(C_k[k]**2,1) / np.sum(X_k[k]**2,1) for k in range(num_sets)] # X_k's variance explained by C_k at the variable level
                
                                       
    if Ell == 0:
        warnings.warn('Warning: The estimated common matrices are zero matrices!')
        C_k = [np.zeros(X_k[k].shape) for k in range(num_sets)]
        PVE_by_Ck_SetLevel = np.zeros(num_sets)
        PVE_by_Ck_VariableLevel = [np.zeros(X_k[k].shape[0]) for k in range(num_sets)]
        
                 
    D_k = [ X_k[k]-C_k[k] for k in range(num_sets) ]
    
                                
    return X_k, C_k, D_k, PVE_by_Ck_SetLevel, PVE_by_Ck_VariableLevel


def select_K_by_GR(eigenv, K_max = None, K_min = None):
    #select the rank, i.e., the number of factors
    # S.C. Ahn, and A.R. Horenstein (2013) EIGENVALUE RATIO TEST FOR THE NUMBER OF FACTORS, Econometrica, 81.    
    
    
    m = len(eigenv)
    
    if K_min is None:
        K_min = 1
    
    if K_max is None or K_max > 0.5*m:
        K_max_star = sum(eigenv >= eigenv.mean())
        K_max = int(np.ceil(min(K_max_star, 0.1*m)))
                            
    if K_max < K_min:
        raise ValueError('In the function select_K_by_GR(), K_min > K_max')


        
    V = np.zeros(K_max+1)      
    V[0]=sum(eigenv[1:])        
    for k in range(1,K_max+1):
        V[k] = V[k-1]-eigenv[k]

    eigenv_star = eigenv[0:(K_max+1)] / V    
    
    GR = np.log(1+eigenv_star[:-1])/np.log(1+eigenv_star[1:])
    
    K = np.argmax(GR[(K_min-1):])+ K_min-1 +1
        
    return K
    

    
def select_K_by_ED(eigenv, K_max = None, K_min = None): 
    #select the rank, i.e., the number of factors
    #Onatski, Alexei. "Determining the number of factors from empirical distribution of eigenvalues." 
    #The Review of Economics and Statistics 92, no. 4 (2010): 1004-1016.
    
    m = len(eigenv)
    
    if K_min is None:
        K_min = 1
    
    if K_max is None or K_max > 0.5*m:
        K_max_star = sum(eigenv >= eigenv.mean())
        K_max = int(np.ceil(min(K_max_star, 0.1*m)))
                            
    if K_max < K_min:
        raise ValueError('In the function select_K_by_ED(), K_min > K_max')
        
    
    
    eigev_diff=eigenv[:K_max]-eigenv[1:(K_max+1)]
    
    K_pre = -1
    j = K_max + 1
    for t in range(100):
        y = eigenv[(j-1):(j+4)]
        x = (j + np.arange(-1,4))**(2/3)  
        lm = LinearRegression()
        lm.fit(x.reshape(5,1),y)
        delta = 2*abs(np.asscalar(lm.coef_))
        index = np.nonzero(eigev_diff >= delta)[0]
        if len(index)==0:
            K = 0
        else:
            K = max(index) + 1
   
        if K_pre == K:
            break
        
        
        K_pre = K
        
        j = K + 1
        
        
    return max(K,K_min)




    
def test_zero_corr(x, y, tail='two', sig_level=0.05):  
    #The studentized test for zero correlation based on the normal approximation 
    #which is proposed by DiCiccio, C. J. and Romano, J. P. (2017), “Robust Permutation Tests For Correlation And Regression Coefficients,” JASA.
    
    '''
    Assume x, y are mean zero.
    '''
    
    if tail in ['two', 'left', 'right']:
        n = len(x)
        rho = x.T @ y / ( norm(x) * norm(y) )
        tau = ( n*(x**2).T @ y**2 / (sum(x**2) * sum(y**2)) )**0.5
        test_stat = n**0.5 * rho / tau
        
        if tail=='two':
            pvalue = 2 * stats.norm.cdf(-abs(test_stat))                
        elif tail=='left':
            pvalue = stats.norm.cdf(test_stat)
        else:
            pvalue = 1 - stats.norm.cdf(test_stat)
        
            
        rejection = (pvalue < sig_level)
               
        return rejection, pvalue
        
    else:
        raise ValueError('The argument "tail" must be one of "two", "left" and "right" .') 
            
    
def decide_alpha_sign(data):            
    global alpha_plus, alpha_minus, I_Delta_ell
    
    sample_size = data.shape[0]
    num_sets = data.shape[1]-1
    
    W_ell = data[:,0]
    Z_kl = data[:,1:].T
    
    
    alpha_jk = np.zeros(int(num_sets*(num_sets-1)/2+2)) #add inf and -inf at the end
    alpha_jk[-1]=np.inf
    alpha_jk[-2]=-np.inf
    
    
    
    t=-1
    for j in range(num_sets-1):
        
        corr_wl_zjl = W_ell.T @ Z_kl[j,:]/ sample_size
        
        for k in range(j+1,num_sets):                        
            t = t+1
            
            corr_wl_zkl = W_ell.T @ Z_kl[k,:]/ sample_size
            
            corr_zjl_zkl = Z_kl[j,:].T @ Z_kl[k,:]/ sample_size
            
            Delta_jk = (corr_wl_zjl + corr_wl_zkl)**2 - 4*corr_zjl_zkl  

            if I_Delta_ell[t]==0:
                alpha_jk[t] = np.inf # indicate no alpha_jk here
            else:
                Delta_jk = max(Delta_jk,0)
                alpha_jk[t] = 0.5 * (corr_wl_zjl + corr_wl_zkl - Delta_jk**0.5)              
            
                   

                
    alpha_plus_bca = min(alpha_jk[alpha_jk>0])
    alpha_minus_bca = max(alpha_jk[alpha_jk<0])
    
    if alpha_plus_bca == np.inf:
        alpha_plus_bca = alpha_plus
    
    if alpha_minus_bca == -np.inf:
        alpha_minus_bca = alpha_minus
        
    return alpha_plus_bca - abs(alpha_minus_bca)
                
 

def CFTtest_for_r_k_star(Z_k, cov_tilde_z_k, alpha=0.05, BootstrapSize=10000):  
    #The CF-T test of Chen, Q. and Fang, Z. (in press). Improved inference on the rank of a matrix. Quantitative Economics
    #Z_k: size r_k * n
    
    r_k, n = Z_k.shape
    
    beta = alpha/10
    
    ind_Boots = 0 #indicating if bootstrapping is used.
    
    #determine r_k_star     
    #Step 1
    P_n, Sigma_n_diag, Q_n_t= sp.linalg.svd(cov_tilde_z_k) # Sigma_n_r12 is a vector
    Sigma_n = np.diag(Sigma_n_diag)
    
    #Step 2: The KP test based on Appendix B of Chen, Q. and Fang, Z. (in press). Improved inference on the rank of a matrix. Quantitative Economics
    r_k_sq = r_k * r_k        
    Omega_n = np.zeros([r_k_sq, r_k_sq])
    for i in range(n):
        temp1 = np.reshape( Z_k[:,i,None] @ Z_k[:,i,None].T - cov_tilde_z_k, [r_k_sq, 1], 'F')
        Omega_n = Omega_n + temp1 @ temp1.T / n 
        
    CFT_rej = 1     
    
    r_null = None
    if r_k == 1:
        r_null = 1
    else:                            
        for r_null in range(1, r_k):     
            kp_rej = 1    
            for r_n_hat in range(1, r_k): 
                P_2n = P_n[:,r_n_hat:]
                Q_2n = Q_n_t.T[:,r_n_hat:]    
                Sigma_2n = Sigma_n[r_n_hat:,r_n_hat:]      
                
                temp1 = np.kron(Q_2n, P_2n)
                temp2 = np.reshape(Sigma_2n, [(r_k-r_n_hat)*(r_k-r_n_hat),1], 'F')
                
                T_nkp = n * temp2.T @ np.linalg.pinv( temp1.T @ Omega_n @ temp1, hermitian=True) @ temp2 #Use numpy 1.17 or above for Use numpy 1.17 or above for hermitian
                
                pvalue = 1 - sp.stats.chi2.cdf(T_nkp, (r_k-r_n_hat)*(r_k-r_n_hat))            
                if pvalue > beta:
                    kp_rej = 0
                    break
                
            if (r_n_hat == (r_k - 1)) and kp_rej:
                 r_n_hat = r_k
                 
            if r_n_hat <= r_null:
                #Step 3
                if ind_Boots == 0:
                    ind_Boots = 1
                    M_n_r12 = np.zeros([BootstrapSize, r_k, r_k])                    
                    np.random.seed(0)                    
                    for B in range(BootstrapSize):
                        index = np.random.choice(n, n)
                        M_n_r12[B] = Z_k[:, index] @ Z_k[:, index].T / n - cov_tilde_z_k
                
                #Step 4
                P_2n = P_n[:,r_n_hat:]
                Q_2n = Q_n_t.T[:,r_n_hat:]     
                
                c_n = [ sum( (sp.linalg.svdvals(P_2n.T @ M_n_r12[B] @ Q_2n)[(r_null-r_n_hat):(r_k-r_n_hat)])**2 ) for B in range(BootstrapSize) ]                
                c_n.sort(reverse=True)#in the descending order
                
                #Step 5
                if  sum(Sigma_n_diag[r_null:]**2) <= c_n[ int( max(np.floor(BootstrapSize*(1-alpha+beta))-1, 0) ) ]:
                    CFT_rej = 0
                    break
                        
        if (r_null == (r_k - 1)) and CFT_rej:
            r_null = r_k
         
    #r_k_star = r_null             
             
    return r_null       
        


    
    
    
    
    
    
    
    
    



