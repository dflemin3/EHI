#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import os
import corner
import emcee
from ehi import mcmcUtils
from statsmodels.stats.proportion import proportion_confint
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (9,8)
mpl.rcParams['font.size'] = 20.0

## for Palatino and other serif fonts use:
#mpl.rc('font',**{'family':'serif'})
#mpl.rc('text', usetex=True)

# Path to data
filename = "../../Data/trappist1WRaymond2007EpsBolmont.h5"

# Whether or not to plot blobs
plotBlobs = True

# Open file
reader = emcee.backends.HDFBackend(filename)

# Compute burnin
tau = reader.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.5*np.min(tau))
print("Burnin, thin:", burnin, thin)

# Load data
chain = reader.get_chain(discard=burnin, flat=True, thin=thin)
tmp = reader.get_blobs(discard=burnin, flat=True, thin=thin)
blobs = []
for bl in tmp:
    blobs.append([bl[ii] for ii in range(len(bl))])
blobs = np.array(blobs)

# Select correct columns
mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43]
blobs = blobs[:,mask]

#14, 15, 16, 17, 18, 19, 20
# Combine!
samples = np.hstack([chain, blobs])

# Define Axis Labels
labels = ["Mass", "SatXUVFrac", "SatXUVTime", "Age", "XUVBeta", "Lum",
          "logLumXUV", "dRGTimee", "dRGTimef", "dRGTimeg", "dRGTimeh",
          "WaterMassb", "WaterMassc", "WaterMassd", "WaterMasse", "WaterMassf", "WaterMassg", "WaterMassh",
          "OxygenMassb", "OxygenMassc", "OxygenMassd", "OxygenMasse",
          "OxygenMassf", "OxygenMassg", "OxygenMassh"]

# Plot lost water content for HZ planets
fig, ax = plt.subplots()

handles = []
inds = [35, 36, 37, 38, 39, 40, 41]

for ind in inds:
    ehi = samples[:,ind] > 0
    ehiEst = np.mean(ehi)
    ehiErrDown, ehiErrUp = proportion_confint(np.sum(ehi), len(ehi),
                                              alpha=0.32, method="agresti_coull")
    print(ehiEst, ehiErrUp - ehiEst, ehiEst - ehiErrDown)
    print()
xxx

colors = ["C%d" % ii for ii in range(len(inds))]
labels = ["TRAPPIST-1 e", "TRAPPIST-1 f", "TRAPPIST-1 g"]
for ii, ind in enumerate(inds):

    # Plot histogram of lost water (all planets started with 20)
    h = ax.hist(20 - samples[:,ind], bins="auto", range=[0, 20], color=colors[ii],
                histtype="step", lw=3, density=True, label=labels[ii])
    h2 = ax.hist(20 - samples[:,ind], bins="auto", range=[0, 20], color=colors[ii],
                 density=True, label="", alpha=0.5)
    handles.append((h[2][0], h2[2][0]))

# Format
ax.set_xlim(2, 16)
ax.set_xlabel("Lost Water [Earth Oceans]", fontsize=24)
ax.set_ylabel("Posterior Density", fontsize=24)
ax.legend(handles, labels, loc="upper right", framealpha=0.0, fontsize=20)

# Save!
fig.savefig("../../Plots/lostWaterPosterior.pdf", bbox_inches="tight", dpi=200)
