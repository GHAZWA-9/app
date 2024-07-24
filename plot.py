
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import ipywidgets as widgets
from IPython.display import display
def interactive_plot(mu_MDE=2.0, sigma=1.0, alpha=0.05, power=0.8):
    x_min = -10
    x_max = max(-x_min, mu_MDE + 5 * sigma)  
    x = np.linspace(x_min, x_max, 1000)
    mu_H0 = 0  # Mean of the null hypothesis distribution
    # Create distributions
    H0_distribution = norm.pdf(x, mu_H0, sigma)
    MDE_distribution = norm.pdf(x, mu_MDE, sigma)
    plt.figure(figsize=(13, 6))
    plt.plot(x, H0_distribution, color='blue', label='H0 Sample Distribution (Assumes no effect)')
    #plt.fill_between(x, H0_distribution, color='teal', alpha=0.3)
    plt.plot(x, MDE_distribution, color='green', label='Alternative Hypothesis Distribution')
 
    plt.axvline(mu_H0, color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(mu_MDE, color='green', linestyle='dashed', linewidth=1)
    # Calculating the critical z-values for alpha and the inverse of power (beta)
    z_alpha = norm.ppf(1 - alpha, mu_H0, sigma)
    beta = 1 - power
    z_beta = norm.ppf(beta, mu_MDE, sigma)
    plt.fill_betweenx(H0_distribution, x, z_alpha, where=(x > z_alpha), color='blue', alpha=0.5, label='Type I Error (α)')
    plt.fill_betweenx(MDE_distribution, x, z_beta, where=(x < z_beta), color='green', alpha=0.5, label='Type II Error β')
    
    plt.title('Properly Powering Your (One-sided) AB Test')
    plt.ylim(0,1)
    plt.legend()
    plt.grid(True)
    plt.show()
# Create interactive widgets
widgets.interact(interactive_plot, MDE=(0.0, 1.0, 0.01), sigma=(0.5, 2.0, 0.1), alpha=(0.01, 0.2, 0.01), power=(0.5, 0.99, 0.01))
##########################################

# This part aim to show that the equal data split between the original and the variant B is the one that minimizes the variance of the estimateur.

# %%
p=0.4
ntot=100
#eps=0.0000000001
def vsmile(p,ntot,alpha):
    return np.sqrt(p*(1-p)*(1/(ntot*(alpha))+1/(ntot*(1-alpha))))
X=np.linspace(0,1,40)
Y=[]
for x in X :
    Y.append(vsmile(p,ntot,x))
#plt.plot(X,Y)
fig, ax = plt.subplots()

ax.plot(X, Y, label='Estimated Volatility')
x_point =0.5
y_point = vsmile(p,ntot,x_point)
ax.plot(x_point, y_point, 'ro')  
ax.set_title('variance in function of data split ')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.ylim(0,0.4)
ax.legend()
# Afficher la figure
plt.show()


########################################################