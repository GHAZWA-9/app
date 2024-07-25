
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

    def get_sample_size (self) :
        #we assume the two population have the same variance which is plausible under H0=sigma=self.baseline(1-self.baseline)
        sigma_2=self.baseline*(1-self.baseline)
        m=1/(self.ratio*(1-self.ratio))*sigma_2*(norm.ppf(1-self.alpha/2, loc=0, scale=1)+norm.ppf(1-self.beta, loc=0, scale=1))**2/(self.baseline*self.mde)**2
        ###add the ratio data split 
    
        return (floor(m))

        
        
    def calculate_duration(self, weekly_traffic):
        sample_size =self.get_sample_size()
        days_required = np.ceil(sample_size / weekly_traffic)
        return days_required

            
    







# %% [markdown]
# This part aim to show that the equal data split between the original and the variant B is the one that minimizes the variance of the estimateur.


# %%
#Unequal allocation between traffic 


