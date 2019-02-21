#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot the marginalized posterior distribution of final water contents, relative to the prior
"""

import numpy as np
import os
import corner
import emcee
from ehi import utils, mcmcUtils
from statsmodels.stats.proportion import proportion_confint
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (9,8)
mpl.rcParams['font.size'] = 16.0

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

### Helper function to load MCMC data ###
def loadData(filename):
    # Open file
    reader = emcee.backends.HDFBackend(filename)

    # Compute burnin
    tau = reader.get_autocorr_time(tol=0)
    burnin = int(2*np.max(tau))
    thin = int(0.5*np.min(tau))

    # Load data
    chain = reader.get_chain(discard=burnin, flat=True, thin=thin)
    tmp = reader.get_blobs(discard=burnin, flat=True, thin=thin)
    blobs = []
    for bl in tmp:
        blobs.append([bl[ii] for ii in range(len(bl))])
    blobs = np.array(blobs)

    # Compute EHI
    ehi = np.array(blobs[:,6] > 0)

    # Ignore dEnvMass column
    mask = np.array([0, 1, 2, 3, 4, 6, 7])
    samples = np.concatenate((chain, blobs[:,mask]), axis=1)

    return samples, ehi
# end function

### Define names, containers ###
dataDir = "../../Data"
titles = ["Delta Function", "Mulders+2015", "Uniform", "Raymond+2007"]
filenames = ["proximaWDelta20EpsBolmont.h5", "proximaLogUniformEpsBolmont.h5",
             "proximaUniformEpsBolmont.h5", "proximaRaymond2007EpsBolmont.h5"]
finalWMF = []
finalTO = []
priorWMF = []
priorTO = []
masses = []
ehis = []

### Load data ###
for filename in filenames:
    res = loadData(os.path.join(dataDir,filename))
    finalWMF.append(res[0][:,10]/ utils.MEarth * utils.MTO / res[0][:,8])
    finalTO.append(res[0][:,10])
    ehis.append(res[1])
    masses.append(res[0][:,8])

# Compute prior distributions
fn = [mcmcUtils.waterPriorDeltaSample, mcmcUtils.waterPriorLogUniformSample,
      mcmcUtils.waterPriorUniformSample, mcmcUtils.waterPriorRaymond2007Sample]
for ii in range(len(filenames)):
    priorTO.append(np.array([fn[ii]() for _ in range(len(masses[ii]))]))
    priorWMF.append(priorTO[ii] / utils.MEarth * utils.MTO / masses[ii])

### Plot ###

# WMF
fig, axes = plt.subplots(ncols=4, figsize=(20,6))

bins = np.logspace(-5, np.log10(0.05), 50)
locs = [(1.3e-5, 5100), (1.3e-5, 400), (1.3e-5, 1525), (1.3e-5, 1395)]
for ii in range(len(titles)):
    axes[ii].hist(finalWMF[ii], bins=bins, histtype="step", lw=3, color="C0",
                  density=False, label="Posterior")
    axes[ii].set_xlabel("Water Mass Fraction")
    axes[ii].set_title(titles[ii])
    axes[ii].set_xscale("log")

    # Plot prior
    axes[ii].hist(priorWMF[ii], bins=bins, histtype="step", lw=2, color="k",
                  ls="--", label="Prior")

    # Print EHI, error estimate
    ehiEst = np.mean(ehis[ii])
    ehiErrDown, ehiErrUp = proportion_confint(np.sum(ehis[ii]), len(ehis[ii]),
                                              alpha=0.32, method="agresti_coull")
    axes[ii].text(*locs[ii], "EHI = $%0.2lf^{+%0.2e}_{-%0.2e}$" % (ehiEst, ehiErrUp-ehiEst, ehiEst-ehiErrDown))

axes[0].set_ylabel("Counts")
axes[1].legend(loc="lower right", framealpha=1)

# Save!
fig.savefig("../../Plots/proximaFinalWMFPDF.pdf", bbox_inches="tight",
            dpi=200)

# TO
fig, axes = plt.subplots(ncols=4, figsize=(20,6))

bins = "auto"
ranges = [(-1, 22), (-1, 10), (-5, 100), (-5, 200)]
locs = [(0, 785), (-0.5, 8200), (0, 1275), (5, 980)]
for ii in range(len(titles)):
    axes[ii].hist(finalTO[ii], bins=bins, histtype="step", lw=3, color="C0",
                  density=False, range=ranges[ii], label="Posterior")
    axes[ii].set_xlabel("Surface Water Content [TO]")
    axes[ii].set_title(titles[ii])
    axes[ii].set_xlim(ranges[ii])

    # Plot prior
    if ii == 0:
        axes[ii].axvline(20, lw=2, color="k", ls="--", label="Prior")
    else:
        axes[ii].hist(priorTO[ii], bins=bins, histtype="step", lw=2, color="k",
                      ls="--", label="Prior", range=ranges[ii])

    # Print EHI, error estimate
    ehiEst = np.mean(ehis[ii])
    ehiErrDown, ehiErrUp = proportion_confint(np.sum(ehis[ii]), len(ehis[ii]),
                                              alpha=0.32, method="agresti_coull")
    axes[ii].text(*locs[ii], "EHI = $%0.2lf^{+%0.2e}_{-%0.2e}$" % (ehiEst, ehiErrUp-ehiEst, ehiEst-ehiErrDown))

axes[0].set_ylabel("Counts")
axes[1].legend(loc="center right", framealpha=1)

# Save!
fig.savefig("../../Plots/proximaFinalTOPDF.pdf", bbox_inches="tight",
            dpi=200)


# Done!
