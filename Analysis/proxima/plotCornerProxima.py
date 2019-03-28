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
mpl.rcParams['font.size'] = 18.0

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

# Path to data
filename = "../../Data/proxima.h5"

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
    ehi = np.array(blobs[:,6] > 0)

    # Estimate error on EHI
    ehiEst = np.mean(ehi)
    ehiErrDown, ehiErrUp = proportion_confint(np.sum(ehi), len(ehi), alpha=0.32, method="agresti_coull")
    print("EHI +/-: %e %e/%e" % (ehiEst, ehiErrUp-ehiEst, ehiEst-ehiErrDown))

    # Ignore dEnvMass column
    mask = np.array([0, 1, 2, 3, 4, 6, 7])
    samples = np.concatenate((chain, blobs[:,mask]), axis=1)

    # Define labels
    labels = ["Mass", "SatXUVFrac", "SatXUVTime", "Age", "XUVBeta", "Lum",
              "logLumXUV", "Porb", "Mass", "dRGTime", "WaterMass",
              "OxygenMass"]

    # Convert RG Time to Myr
    samples[:,9] = samples[:,9]/1.0e6

    # Make luminosity units more palatable
    samples[:,5] = samples[:,5]*1.0e3
else:
    samples = chain
    labels = [r"$m_{\star}$", r"$f_{sat}$", r"$t_{sat}$", r"$t_{\star}$", r"$\beta_{XUV}$"]

# Plot!
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], labels=labels,
                    show_titles=True, title_kwargs={"fontsize": 12})

# Save!
fig.savefig("../../Plots/proximaCorner.pdf", bbox_inches="tight", dpi=200)

# Done!
