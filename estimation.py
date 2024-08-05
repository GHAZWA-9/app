"""# Write code to generate synthetic AB test data. Converted visits, conversions, revenue…
# Write lib to compute test statistic from this data. Check how others do and compare results.
# This will be the basis for future work.
# N: # de vsiteurs qui sont ciblés par le test
# : nombre de variation (=2)
# : répartition entre les variations (50%-50%)
# : taux de conversion de base du goal (10%)
# : MDE - effet minimal que l’on veut détecter (5% relatif)
# 2 variant case : number of the variant in function of MDE estimate the minimal number of variant users
# alpha: risque de type I
# beta: risque de type II
# H0: pas de différences entre l’originale et la variation
"""
import math as m
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class ABTEST:
    """sample size calculator class """

    def __init__(self, nv =2, mde=0.05 , alpha=0.05, beta =0.2, ctr1=0.1, traffic=1000 ) -> None:
        self.nbre_va = nv
        self.mde = mde
        self.alpha = alpha
        self.baseline = ctr1
        self.beta = beta
        # self.ratio=r
        self.traffic = traffic

    def get_sample_size(self, click:str) -> int :
        # we assume the two population have the same variance which is plausible under H0=sigma=self.baseline(1-self.baseline)

        sigma_2 = self.baseline * (1 - self.baseline)
        # S=(self.ratio*(1-self.ratio))
        if click == "One-sided Test":

            m = 2 * sigma_2 * (norm.ppf(1 - self.alpha, loc=0, scale=1) + norm.ppf(1 - self.beta, loc=0, scale=1)) ** 2 / (self.baseline * self.mde) ** 2

        else:
            m=2 * sigma_2 * (norm.ppf(1 - self.alpha/2, loc=0, scale=1) + norm.ppf(1 - self.beta, loc=0, scale=1)) ** 2 / (self.baseline * self.mde) ** 2
        ###add the ratio data split

        return m.floor(m)

    def calculate_duration(self, weekly_traffic, click):
        sample_size = self.get_sample_size(click)
        days_required = np.ceil(sample_size / weekly_traffic)
        return days_required

    def plot_distributions(self, sample_size : int) -> None: 
        control_mean = self.baseline
        treatment_mean = self.baseline * (1 + self.mde)
        control_std = np.sqrt(control_mean * (1 - control_mean) / sample_size)
        treatment_std = np.sqrt(treatment_mean * (1 - treatment_mean) / sample_size)

        control_group = np.random.normal(control_mean, control_std, 10000)
        treatment_group = np.random.normal(treatment_mean, treatment_std, 10000)

        plt.figure(figsize=(10, 5))
        plt.hist(control_group, bins=50, alpha=0.5, label="Control")
        plt.hist(treatment_group, bins=50, alpha=0.5, label="Treatment")
        plt.legend(loc="upper right")
        plt.xlabel("Conversion Rate")
        plt.ylabel("Frequency")
        plt.title("Distributions of Conversion Rates")
        return plt

    def calculate_mde(self) ->float : 
        sigma_2 = self.baseline * (1 - self.baseline)
        s = (
            (
                norm.ppf(1 - self.alpha, loc=0, scale=1)
                + norm.ppf(1 - self.beta, loc=0, scale=1)
            )
            ** 2
            * 2
            * sigma_2
            / self.traffic
        )
        return np.sqrt(s) * 100


# %% [markdown]
# This part aim to show that the equal data split between the original and the variant B is the one that minimizes the variance of the estimateur.


# %%
# Unequal allocation between traffic
