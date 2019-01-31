#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (9,8)
mpl.rcParams['font.size'] = 26.0

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

# Read in evolutionary tracks
data = np.load("../../Data/proximaEvol.npz")
nsamples = len(data["Luminosity"])

### Plot Lum, LumXUV, radius evolution and compare to observations ###

fig, axes = plt.subplots(ncols=3, figsize=(21,6))

for ii in range(nsamples):

    # Left: lum
    axes[0].plot(data["time"][ii], data["Luminosity"][ii], alpha=0.1, color="k",
                 lw=2, zorder=1)

    # Middle: lumLUX
    axes[1].plot(data["time"][ii], data["LXUVStellar"][ii], alpha=0.3, color="k",
                 lw=2)

    # Middle: lumLUX
    axes[2].plot(data["time"][ii], data["Radius"][ii], alpha=0.1, color="k",
                 lw=2)

# Plot constraints, format

# Luminosity from Demory+2009
x = np.linspace(0, 7.5e9, 100)
axes[0].fill_between(x, 0.00165-0.00015, 0.00165+0.00015, color="C0",
                     alpha=0.3, zorder=0)
axes[0].axhline(0.00165, color="C0", lw=2, ls="--", zorder=2)

axes[0].set_ylabel("Luminosity [L$_{\odot}$]", fontsize=25)
axes[0].set_xlabel("Time [yr]", fontsize=25)
axes[0].set_yscale("log")
axes[0].set_xscale("log")

# XUV Luminosity from Ribas+2016
axes[1].fill_between(x, 10**(-6.36-0.3), 10**(-6.36+0.3), color="C0",
                     alpha=0.3, zorder=0)
axes[1].axhline(10**-6.36, color="C0", lw=2, ls="--", zorder=2)

axes[1].set_ylabel("XUV Luminosity [L$_{\odot}$]", fontsize=25)
axes[1].set_xlabel("Time [yr]", fontsize=25)
axes[1].set_yscale("log")
axes[1].set_xscale("log")

# Radius from Anglada-Escude+2016
axes[2].fill_between(x, 0.12, 0.162, color="C0", alpha=0.3, zorder=0)
axes[2].axhline(0.141, color="C0", lw=2, ls="--", zorder=2)

axes[2].set_ylabel("Radius [R$_{\odot}$]", fontsize=25)
axes[2].set_xlabel("Time [yr]", fontsize=25)
axes[2].set_xscale("log")

fig.savefig("../../Plots/proximaEvol.png", bbox_inches="tight", dpi=200)

### Plot runaway greenhouse HZ limit ###

fig, ax = plt.subplots(figsize=(7,6))

for ii in range(nsamples):

    # Left: lum
    ax.plot(data["time"][ii], data["HZLimRunaway"][ii], alpha=0.1,
            color="k", lw=2, zorder=1)

# Format, plot Proxima b's current semi-major axis from Anglada-Escude+2016
x = np.linspace(0, 7.5e9, 100)
ax.fill_between(x, 0.0434, 0.0526, color="C0", alpha=0.3, zorder=0)
ax.axhline(0.0485, color="C0", lw=2, ls="--", zorder=2)

ax.set_xscale("log")
ax.set_xlabel("Time [yr]", fontsize=25)
ax.set_ylabel("Distance [AU]", fontsize=25)

fig.savefig("../../Plots/proximaHZLimEvol.png", bbox_inches="tight", dpi=200)

### Plot surface water mass, oxygen buildup amounts ###

fig, axes = plt.subplots(ncols=2, figsize=(14,6))

for ii in range(nsamples):

    # Left: water
    axes[0].plot(data["time"][ii], data["SurfWaterMass"][ii], alpha=0.1, color="k",
                 lw=2, zorder=1)

    # right: O2
    axes[1].plot(data["time"][ii], data["OxygenMass"][ii], alpha=0.1, color="k",
                 lw=2)

# Format
axes[0].set_ylabel("Surface Water [TO]", fontsize=25)
axes[0].set_xlabel("Time [yr]", fontsize=25)
axes[0].set_xscale("log")

axes[1].set_ylabel("O$_{2}$ [bars]", fontsize=25)
axes[1].set_xlabel("Time [yr]", fontsize=25)
axes[1].set_yscale("log")
axes[1].set_xscale("log")

fig.savefig("../../Plots/proximaBEvol.png", bbox_inches="tight", dpi=200)
# Done!
