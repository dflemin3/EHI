#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize initial water inventories under different assumed priors.

Priors:

- Uniform over [0, 100] TO
- Delta function at 20 TO
- loguniform over [1.0e-5, 5.0e-2] based on Ciesla+2015, Mulders+2015 in situ models
- Fit to Raymond+2004 high resolution simulations of planets in the HZ of a Sun-like star

@author: David P. Fleming, University of Washington, 2019
"""

import numpy as np
import os
from scipy.stats import norm
import matplotlib as mpl
import matplotlib.pyplot as plt

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (9,8)
mpl.rcParams['font.size'] = 26.0

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

# Set RNG Seed
np.random.seed(42)

### Fit Raymond+2007 water mass fractions

# Raymond+2007 water mass fractions
ray = np.log10([2.6e-3, 8.4e-3, 9.1e-3, 8.3e-3, 5.5e-3, 1.2e-2, 7.2e-3, 6.7e-3, 3.8e-3,
       9.3e-4, 8.6e-3, 6e-3, 1.8e-2, 7.1e-3, 2e-2])

# Fit Gaussian
mu, std = norm.fit(ray)
print("Fitted mu, std for log10 Raymond+2007 WMF:", mu, std)

# Examine fit
fig, ax = plt.subplots()

ax.hist(ray, bins="auto", histtype="step", lw=3, color="C0", label="Data",
        density=True)
ax.hist(norm.rvs(mu, std, size=10000), bins="auto", histtype="step", lw=3,
        color="C1", label="Fit", density=True)

ax.set_ylabel("Counts")
ax.set_xlabel("Log10 Water Mass Fraction")
ax.legend(loc="best")

fig.savefig("../Plots/raymond2007Fit.png", bbox_inches="tight", dpi=200)

### Examine Priors ###

# Define constants
nsamples = 10000
MTO = 1.4e24 # Mass of all of Earth's water in g
MEarth = 5.972e27 # Mass of Earth in g

### Sample from priors for surface water mass in terrestrial oceans (TO) ###

# Assume a 1 Mearth planet for simplicity
uni = np.random.uniform(low=0, high=100, size=nsamples)
logMulders = 10**np.random.uniform(low=np.log10(1.0e-5), high=np.log10(5.0e-2), size=nsamples) * MEarth / MTO
delta = 20
ray2007 = 10**norm.rvs(mu, std, size=nsamples) * MEarth / MTO

# Plot histograms in terms of TO

fig, ax = plt.subplots()

ax.hist(uni, lw=3, histtype="step", color="C0", label="Uniform", bins="auto",
        range=[0,105])
ax.hist(logMulders, lw=3, histtype="step", color="C1", label="LogUniform",
        bins="auto", range=[0,105])
ax.hist(ray2007, lw=3, histtype="step", color="k", label="Raymond+2007",
        bins="auto", range=[0, 105])
ax.axvline(delta, lw=3, color="C2", label="Delta")

ax.set_xlabel("Initial Water Inventory [TO]")
ax.set_ylabel("Counts")
ax.legend(loc="best")

fig.savefig("../Plots/toPriorHist.png", bbox_inches="tight", dpi=200)

### Perform the same as above, but in terms of water mass fraction ###

# Assume a 1 Mearth planet for simplicity
uni = np.random.uniform(low=0, high=100, size=nsamples) * MTO / MEarth
logMulders = 10**np.random.uniform(low=np.log10(1.0e-5), high=np.log10(5.0e-2), size=nsamples)
delta = 20 * MTO / MEarth
ray2007 = 10**norm.rvs(mu, std, size=nsamples)

fig, ax = plt.subplots()

bins = np.logspace(-5, -1, 50)
ax.hist(uni, lw=3, histtype="step", color="C0", label="Uniform", bins=bins)
ax.hist(logMulders, lw=3, histtype="step", color="C1", label="LogUniform", bins=bins)
ax.hist(ray2007, lw=3, histtype="step", color="k", label="Raymond+2007", bins=bins)
ax.axvline(delta, lw=3, color="C2", label="Delta")

ax.set_xscale("log")
ax.set_xlabel("Water Mass Fraction")
ax.set_ylabel("Counts")
ax.legend(loc="best")

fig.savefig("../Plots/waterMassFracPriorHist.png", bbox_inches="tight", dpi=200)

# Done!
