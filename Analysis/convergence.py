#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (9,8)
mpl.rcParams['font.size'] = 18.0

## for Palatino and other serif fonts use:
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

path = "../Sims/Convergence/"
dirs = ["Eta1", "Eta01", "Eta001", "Eta0001"]
labels = [r"$\eta = 1$", r"$\eta = 10^{-1}$", r"$\eta = 10^{-2}$",
          r"$\eta = 10^{-3}$"]
colors = ["C%d" % ii for ii in range(len(dirs))]
planets = ["b", "c", "d", "e", "f", "g", "h"]

# Plot stellar convergence of HZ lim, LXUV
fig, axes = plt.subplots(ncols=2, figsize=(10, 6))

# Load in "truth"
true = np.genfromtxt(os.path.join(path, "Eta00001", "Trappist.star.forward"), delimiter=" ")
trueLXUV = true[:,2]
trueHZ = true[:,6]

# Loop over dirs contain same simulation, but different integration scaling factors
for ii, dir in enumerate(dirs):

    # Load data
    data = np.genfromtxt(os.path.join(path, dir, "Trappist.star.forward"), delimiter=" ")

    # Left: LXUV
    axes[0].plot(data[:,0], np.fabs(data[:,2] - trueLXUV)/trueLXUV, lw=2, color=colors[ii], label=labels[ii])

    # Right: HZ limt
    axes[1].plot(data[:,0], np.fabs(data[:,6] - trueHZ)/trueHZ, lw=2, color=colors[ii], label=labels[ii])

# Format!
axes[0].set_xlabel("Time [yr]")
axes[0].set_xlim(data[1,0], data[-1,0])
axes[0].set_xscale("log")
axes[0].set_ylabel("LXUV Relative Error")
axes[0].set_yscale("log")
axes[0].legend(loc="best", fontsize=10)

axes[1].set_xlabel("Time [yr]")
axes[1].set_xlim(data[1,0], data[-1,0])
axes[1].set_xscale("log")
axes[1].set_ylabel("HZ Lim Relative Error")
axes[1].set_yscale("log")

# Save!
fig.tight_layout()
fig.savefig("../Plots/Convergence/trappist1StarConv.png", dpi=200, bbox_inches="tight")

# For each planet, plot convergence of SurfWaterMass, Oxygen
for planet in planets:

    # Create figure
    fig, axes = plt.subplots(ncols=2, figsize=(10, 6))

    # Load truth
    true = np.genfromtxt(os.path.join(path, "Eta00001", "Trappist.trappist1%s.forward" % planet), delimiter=" ")

    # Loop over dirs contain same simulation, but different integration scaling factors
    for ii, dir in enumerate(dirs):

        # Load data
        data = np.genfromtxt(os.path.join(path, dir, "Trappist.trappist1%s.forward" % planet), delimiter=" ")

        # Left: water
        axes[0].plot(data[:,0], np.fabs(data[:,2] - true[:,2])/true[:,2], lw=2, color=colors[ii], label=labels[ii])

        # Right: oxygen
        axes[1].plot(data[:,0], np.fabs(data[:,1] - true[:,1])/true[:,1], lw=2, color=colors[ii], label=labels[ii])

    # Format figure, save
    axes[0].set_xlabel("Time [yr]")
    axes[0].set_xlim(data[1,0], data[-1,0])
    axes[0].set_xscale("log")
    axes[0].set_ylabel("Water Content Relative Error")
    axes[0].set_yscale("log")
    axes[0].legend(loc="best", fontsize=10)

    axes[1].set_xlabel("Time [yr]")
    axes[1].set_xlim(data[1,0], data[-1,0])
    axes[1].set_xscale("log")
    axes[1].set_ylabel("Oxygen Mass Relative Error")
    axes[1].set_yscale("log")

    fig.tight_layout()
    fig.savefig(os.path.join("../Plots/Convergence/","trappist1%sConv.png" % planet), dpi=200, bbox_inches="tight")
