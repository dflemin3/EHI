#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot both true and AP corner plots, and find diffence in derived parameters.
"""

import numpy as np
import os
import corner
import emcee
from statsmodels.stats.proportion import proportion_confint
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (9,8)
mpl.rcParams['font.size'] = 18.0

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

### Load true data ###

# Path to data
trueFilename = "../../Data/proxima.h5"

# Open file
trueReader = emcee.backends.HDFBackend(trueFilename)

# Compute burnin
trueTau = trueReader.get_autocorr_time(tol=0)
trueBurnin = int(2*np.max(trueTau))
trueThin = int(0.5*np.min(trueTau))
print("True burnin, thin:", trueBurnin, trueThin)

# Load data
trueSamples = trueReader.get_chain(discard=trueBurnin, flat=True, thin=trueThin)

### Load AP data ###

# Path to data
apFilename = "../../Data/apRun3.h5"

# Open file
apReader = emcee.backends.HDFBackend(apFilename)

# Calculate burnin
apTau = apReader.get_autocorr_time(tol=0)
apBurnin = int(2*np.max(apTau))
apThin = int(0.5*np.min(apTau))
print("AP burnin, thin:", apBurnin, apThin)

# Load data
apSamples = apReader.get_chain(discard=apBurnin, flat=True, thin=apThin)

### Plot! ###

fig, ax = plt.subplots(figsize=(7,6))

# Plot params
bins = 20
labels = ["Mass", "SatXUVFrac", "SatXUVTime", "Age", "XUVBeta"]

# Plot GP approximation to the joint distribution
fig = corner.corner(apSamples, quantiles=[0.16, 0.5, 0.84], labels=labels,
                    show_titles=False, bins=bins, plot_density=False,
                    plot_contours=True, plot_datapoints=False,
                    label="approxposterior", color="dodgerblue", lw=2)

fig = corner.corner(trueSamples, quantiles=[0.16, 0.5, 0.84], labels=labels,
                    show_titles=False, bins=bins, plot_density=True,
                    plot_contours=False, plot_datapoints=False,
                    label="True", color="k", fig=fig, lw=2)

# Add legend
fig.axes[1].text(0.13, 0.55, "True", fontsize=26, color="k", zorder=99)
fig.axes[1].text(0.13, 0.375, "approxposterior", fontsize=26, color="dodgerblue",
                 zorder=99)

# Save!
fig.savefig("../../Plots/proximaCornerOverlap.pdf", bbox_inches="tight", dpi=200)

# Done!
