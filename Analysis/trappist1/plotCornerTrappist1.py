#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
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
mpl.rcParams['font.size'] = 15.0

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

# Path to data
filename = "../../Data/trappist1WRaymond2007EpsBolmont.h5"

# Whether or not to plot blobs
plotBlobs = False

# Open file
reader = emcee.backends.HDFBackend(filename)

# Compute burnin
tau = reader.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.5*np.min(tau))
print("Burnin, thin:", burnin, thin)

# Load data
chain = reader.get_chain(discard=burnin, flat=True, thin=thin)
if plotBlobs:
    tmp = reader.get_blobs(discard=burnin, flat=True, thin=thin)
    blobs = []
    for bl in tmp:
        blobs.append([bl[ii] for ii in range(len(bl))])
    blobs = np.array(blobs)

    # Select correct columns
    mask = [0, 1, 19, 20, 21, 22, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
    blobs = blobs[:,mask]

    # Combine!
    samples = np.hstack([chain, blobs])

    # Define Axis Labels
    labels = ["Mass", "SatXUVFrac", "SatXUVTime", "Age", "XUVBeta", "Lum",
              "logLumXUV", "dRGTimee", "dRGTimef", "dRGTimeg", "dRGTimeh",
              "WaterMassb", "WaterMassc", "WaterMassd", "WaterMasse", "WaterMassf", "WaterMassg", "WaterMassh",
              "OxygenMassb", "OxygenMassc", "OxygenMassd", "OxygenMasse",
              "OxygenMassf", "OxygenMassg", "OxygenMassh"]

    # Convert RG Time to Myr
    samples[:,7] = samples[:,7]/1.0e6
    samples[:,8] = samples[:,8]/1.0e6
    samples[:,9] = samples[:,9]/1.0e6
    samples[:,10] = samples[:,10]/1.0e6

    # Make luminosity units more palatable
    samples[:,5] = samples[:,5]*1.0e3
else:
    # Just consider stellar data
    samples = chain
    labels = [r"$m_{\star}$ [M$_{\odot}$]", r"$f_{sat}$",
              r"$t_{sat}$ [$\log_{10}$(Gyr)]", r"Age [Gyr]", r"$\beta_{XUV}$"]

# Plot!
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], labels=labels,
                    show_titles=True, title_kwargs={"fontsize": 12})

# Save!
fig.savefig("../../Plots/trappist1CornerWRaymond2007EpsBolmont.pdf", bbox_inches="tight", dpi=200)

# Done!
