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
filename = "../../Data/apRun3.h5"

# Open file
reader = emcee.backends.HDFBackend(filename)

# Calculate burnin
tau = reader.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.5*np.min(tau))
print("Burnin, thin:", burnin, thin)

# Load data and plot!
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)

labels = [r"$m_{\star}$", r"$f_{sat}$", r"$t_{sat}$", r"$t_{\star}$", r"$\beta_{XUV}$"]
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], labels=labels,
                    show_titles=True, title_kwargs={"fontsize": 12})

# Save!
fig.savefig("../../Plots/proximaCornerAP.pdf", bbox_inches="tight", dpi=200)

# Done!
