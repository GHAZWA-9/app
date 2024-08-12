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
    plt.plot(
        x,
        H0_distribution,
        color="blue",
        label="H0 Sample Distribution (Assumes no effect)",
    )
    # plt.fill_between(x, H0_distribution, color='teal', alpha=0.3)
    plt.plot(
        x, MDE_distribution, color="green", label="Alternative Hypothesis Distribution"
    )

    plt.axvline(mu_H0, color="blue", linestyle="dashed", linewidth=1)
    plt.axvline(mu_MDE, color="green", linestyle="dashed", linewidth=1)
    # Calculating the critical z-values for alpha and the inverse of power (beta)
    z_alpha = norm.ppf(1 - alpha, mu_H0, sigma)
    beta = 1 - power
    z_beta = norm.ppf(beta, mu_MDE, sigma)
    plt.fill_betweenx(
        H0_distribution,
        x,
        z_alpha,
        where=(x > z_alpha),
        color="blue",
        alpha=0.5,
        label="Type I Error (α)",
    )
    plt.fill_betweenx(
        MDE_distribution,
        x,
        z_beta,
        where=(x < z_beta),
        color="green",
        alpha=0.5,
        label="Type II Error β",
    )

    plt.title("Properly Powering Your (One-sided) AB Test")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()


# Create interactive widgets
widgets.interact(
    interactive_plot,
    MDE=(0.0, 1.0, 0.01),
    sigma=(0.5, 2.0, 0.1),
    alpha=(0.01, 0.2, 0.01),
    power=(0.5, 0.99, 0.01),
)
##########################################
# This part aim to show that the equal data split between the original and the variant B is the one that minimizes the variance of the estimateur.

# %%
p = 0.4
ntot = 100


# eps=0.0000000001
def vsmile(p, ntot, alpha):
    return np.sqrt(p * (1 - p) * (1 / (ntot * (alpha)) + 1 / (ntot * (1 - alpha))))


X = np.linspace(0, 1, 40)
Y = []
for x in X:
    Y.append(vsmile(p, ntot, x))
# plt.plot(X,Y)
fig, ax = plt.subplots()

ax.plot(X, Y, label="Estimated Volatility")
x_point = 0.5
y_point = vsmile(p, ntot, x_point)
ax.plot(x_point, y_point, "ro")
ax.set_title("variance in function of data split ")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.ylim(0, 0.4)
ax.legend()
# Afficher la figure
plt.show()


########################################################


####################################################################


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

"""Power in function of MDE """

# Define the range of x values
x = np.linspace(0, 0.6, 100)
ratio = 0.5
baseline = 0.5
n = 1000  # total sample size
alpha = 0.05


# Compute the CDF values for the standard normal distribution (mean=0, std=1)
def quant(n, ratio, baseline, mde, alpha):
    m = np.sqrt(n * ratio * (1 - ratio)) * baseline * mde - norm.ppf(
        1 - alpha / 2, loc=0, scale=1
    )
    return m


cdf_values = []
for d in x:
    cdf_values.append(
        norm.cdf(quant(n, ratio, baseline, d, alpha), loc=0, scale=1) * 100
    )

# Plot the CDF
plt.figure(figsize=(10, 6))
plt.plot(x, cdf_values, color="blue")
# plt.title('Cumulative Distribution Function (CDF) of a Standard Normal Distribution')
plt.xlabel("MDE")
plt.ylabel("Power %")
# plt.legend()
plt.grid(True)
plt.show()

###################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define the range of x values
x = np.linspace(0, 50000, 100)
ratio = 0.5
p = baseline = 0.5
alpha = 0.05
mde = 0.1


# Compute the CDF values for the standard normal distribution (mean=0, std=1)
def quant(n, ratio, baseline, mde, alpha):
    m = np.sqrt(n * ratio * (1 - ratio)) * baseline * mde - norm.ppf(
        1 - alpha / 2, loc=0, scale=1
    )
    return m


cdf_values = cdf_values1 = []
cdf_values1 = []
for d in x:
    cdf_values.append(
        norm.cdf(quant(d, ratio, baseline, mde, alpha), loc=0, scale=1) * 100
    )
    cdf_values1.append(
        norm.cdf(quant(d, ratio, baseline, mde, 0.01), loc=0, scale=1) * 100
    )

# Plot the CDF
plt.figure(figsize=(10, 6))
plt.plot(x, cdf_values, label="a=0.05", color="blue")
plt.plot(x, cdf_values1, label="a=0.01", color="red")

# plt.title('Cumulative Distribution Function (CDF) of a Standard Normal Distribution')
plt.xlabel("N")
plt.ylabel("Power %")
plt.legend()
plt.grid(True)
plt.show()


# NO FREE LUNCH, decreasing type 1 ERROR (reject wrongly H0) DECREASES statistical power(detect a true significant effect)


################################################
#
def estimateSZ(p, mde, r, alpha, beta):
    # we assume the two population have the same variance which is plausible under H0=sigma=self.baseline(1-self.baseline)
    sigma_2 = p * (1 - p)
    m = (
        1
        / (r * (1 - r))
        * sigma_2
        * (norm.ppf(1 - alpha / 2, loc=0, scale=1) + norm.ppf(1 - beta, loc=0, scale=1))
        ** 2
        / (p * mde) ** 2
    )
    ###add the ratio data split

    return floor(m)
