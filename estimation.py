# %% [markdown]
# Write code to generate synthetic AB test data. Converted visits, conversions, revenue…
# 
# Write lib to compute test statistic from this data. Check how others do and compare results.
# 
# This will be the basis for future work.
# N: # de visiteurs qui sont ciblés par le test
# : nombre de variation (=2)
# : répartition entre les variations (50%-50%)
# : taux de conversion de base du goal (10%)
# : MDE - effet minimal que l’on veut détecter (5% relatif)
# #2 variant case : number of the variant in function of MDE estimate the minimal number of variant users
# alpha: risque de type I
# beta: risque de type II
# 
# H0: pas de différences entre l’originale et la variation

# %%
import numpy as np
import pandas as pd
from scipy import stats  
from scipy.stats import norm
from math import *
class ABTEST :

    def __init__(self, nv=2,mde=0.05,alpha=0.05,beta=0.2,ctr1=0.1,r=0.5):
        self.nbre_va = nv
        self.mde=mde
        self.alpha=alpha 
        self.baseline=ctr1
        self.beta=beta 
        self.ratio=r

    def get_sample_size (self,p=0.5) :
        #we assume the two population have the same variance which is plausible under H0=sigma=self.baseline(1-self.baseline)
        sigma_2=self.baseline*(1-self.baseline)
        m=1/(self.ratio*(1-self.ratio))*sigma_2*(norm.ppf(1-self.alpha/2, loc=0, scale=1)+norm.ppf(1-self.beta, loc=0, scale=1))**2/(self.baseline*self.mde)**2
        ###add the ratio data split 
    
        return (floor(m))
    def get_weekly_traffic (self):

        """  
        read the weekly traffic from the user
           
        """
        
        
    """def get_duration ():"""

            
    #Power analysis method 
    #look at sensitivity to beta and delta given some input data (simulated )








# %% [markdown]
# This part aim to show that the equal data split between the original and the variant B is the one that minimizes the variance of the estimateur.

# %% [markdown]
# Seeing the behaviour of 1-beta in function  OF MDE (CURVE 1)
# seeing the behaviour of power in function of N (for different alpha ) (curve 2)
# 


# %%
#Unequal allocation between traffic 
#to finish

# %%


# %%
import numpy as np
import scipy.stats as stats
import statsmodels.stats.power as smp
from pingouin import power_ttest2n
from statsmodels.stats.weightstats import ztest as ztest

def generate_data(n1, p1, n2, mde):
    """
    Génère deux échantillons de données avec des tailles et des variances spécifiées.
    
    Arguments:
    n1 -- Taille de l'échantillon 1
    var1 -- Variance de l'échantillon 1
    n2 -- Taille de l'échantillon 2
    var2 -- Variance de l'échantillon 2
    
    Retourne:
    data1 --
    data2 --
    """
    data1 = np.random.binomial(1, p1, int(n1))
    data2 = np.random.binomial(1, p1*(1+mde),int(n2) )
    
    return data1, data2

def perform_z_test(data1, data2):
    """
    
    Arguments:
    data1 -- base sample
    data2 -- variant sample
    
    Retourne:
    t_stat -- t-test
    p_value
    """
    z_stat, p_value = ztest (data1, data2, value=0) 
    return z_stat, p_value


# Parameters

    
#n = 1000
p1 =0.4  
mde=0.1
p2=0.4*(1+mde)
r=0.5
beta=0.2
n=estimateSZ (p1,mde,r,alpha,beta)
print(n)
# simulation
data1, data2 = generate_data(r*n, p1, (1-r)*n,mde )

# T-test
t_stat, p_value = perform_z_test(data1, data2)
alpha = 0.05
num_simulations=1000



    

# %%

def compute_test_power(alpha=0.05,num_simulations=10000):
    powers = 0
    for _ in range(num_simulations):
        control_data,treatment_data= generate_data(r*n, p1, (1-r)*n,mde )
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        control_var = np.std (control_data)**2
        treatment_var = np.std(treatment_data)**2
        z_stat, p_value = perform_z_test(control_data,treatment_data)
        if abs(z_stat) > norm.ppf(1 - alpha / 2):
            powers += 1
    power_rate = powers / num_simulations
    return power_rate
power_rate = compute_test_power(alpha,num_simulations=10000)
print(f"Power Rate: {power_rate:.4f}")

# %% [markdown]
# Simulating user data with intracorrelation.

# %%
import numpy as np
from scipy.stats import norm, truncnorm, poisson, bernoulli, ttest_ind
# Set parameters
px = 0.6
delta = 0.05
sigma = 0.175
K = 5000
lambda_sessions = 3
alpha = 0.05
power = 0.80
num_simulations = 1000
k_per_arm = 448
def truncated_normal(mean, sd, lower, upper):
    a, b = (lower - mean) / sd, (upper - mean) / sd
    return truncnorm(a, b, loc=mean, scale=sd)
def simulate_user_data(mu, num_users):
    session_counts = poisson.rvs(mu=lambda_sessions, size=num_users)
    #ession_counts = np.array(1*num_users)
    pis = truncated_normal(mu, sigma, mu - 2 * sigma, mu + 2 * sigma).rvs(size=num_users)
    conversions = [bernoulli.rvs(pi, size=n).sum() for pi, n in zip(pis, session_counts)]
    return conversions, session_counts
def delta_method_var(mu, sigma, N):
    return sigma**2 / N
def run_simulation():
    type_I_errors = 0
    powers = 0
    for _ in range(num_simulations):
        # Null hypothesis simulation
        control_conversions, control_sessions = simulate_user_data(px, k_per_arm)
        treatment_conversions, treatment_sessions = simulate_user_data(px, k_per_arm)
        control_mean = np.mean(control_conversions)
        treatment_mean = np.mean(treatment_conversions)
        control_var = delta_method_var(px, sigma, np.mean(control_sessions))
        treatment_var = delta_method_var(px, sigma, np.mean(treatment_sessions))
        t_stat = (treatment_mean - control_mean) / np.sqrt(treatment_var + control_var)
        if abs(t_stat) > norm.ppf(1 - alpha / 2):
            type_I_errors += 1
        # Alternative hypothesis simulation
        control_conversions, control_sessions = simulate_user_data(px, k_per_arm)
        treatment_conversions, treatment_sessions = simulate_user_data(px + delta, k_per_arm)
        control_mean = np.mean(control_conversions)
        treatment_mean = np.mean(treatment_conversions)
        control_var = delta_method_var(px, sigma, np.mean(control_sessions))
        treatment_var = delta_method_var(px + delta, sigma, np.mean(treatment_sessions))
        t_stat = (treatment_mean - control_mean) / np.sqrt(treatment_var + control_var)
        if abs(t_stat) > norm.ppf(1 - alpha / 2):
            powers += 1
    type_I_error_rate = type_I_errors / num_simulations
    power_rate = powers / num_simulations
    return type_I_error_rate, power_rate
type_I_error_rate, power_rate = run_simulation()
print(f"Type I Error Rate: {type_I_error_rate:.4f}")
print(f"Power Rate: {power_rate:.4f}")



