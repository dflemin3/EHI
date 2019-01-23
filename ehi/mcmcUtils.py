#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
docs
"""

import vplot as vpl
import numpy as np
import emcee
import subprocess
import re
import os, sys
import random
import argparse
from . import utils


__all__ = ["FunctionWrapper", "LnLike", "GetEvol", "RunMCMC"]

class FunctionWrapper(object):
    """"
    A simple function wrapper class. Stores :py:obj:`args` and :py:obj:`kwargs` and
    allows an arbitrary function to be called with a single parameter :py:obj:`x`
    """

    def __init__(self, f, *args, **kwargs):
        """
        """

        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        """
        """

        return self.f(x, *self.args, **self.kwargs)
# end class

def LnLike(x, **kwargs):
    """
    loglikelihood function: runs VPLanet simulation!
    """

    # Get the current vector
    dMass, dSatXUVFrac, dSatXUVTime, dStopTime, dXUVBeta = x
    dSatXUVFrac = 10 ** dSatXUVFrac
    dStopTime *= 1.e9
    dOutputTime = dStopTime
    dSatXUVTime = 10 ** dSatXUVTime

    # Get the prior probability
    lnprior = kwargs["LnPrior"](x, **kwargs)
    if np.isinf(lnprior):
        return -np.inf, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    # Must have orbital period, error, for planet
    porb = kwargs.get("PORB")
    porbSig = kwargs.get("PORBSIG")
    try:
        lum = kwargs.get("LUM")
        lumSig = kwargs.get("LUMSIG")
    except KeyError:
        lum = None
        lumSig = None
    try:
        lumXUV = kwargs.get("LUMXUV")
        lumXUVSig = kwargs.get("LUMXUVSIG")
    except KeyError:
        lumXUV = None
        lumXUVSig = None

    # Get strings containing VPLanet input filex (they must be provided!)
    try:
        planet_in = kwargs.get("PLANETIN")
        star_in = kwargs.get("STARIN")
        vpl_in = kwargs.get("VPLIN")
    except KeyError as err:
        print("ERROR: Must supply PLANETIN, STARIN, VPLIN.")
        raise

    # Get PATH
    try:
        PATH = kwargs.get("PATH")
    except KeyError as err:
        print("ERROR: Must supply PATH.")
        raise

    # Randomize file names
    sysName = 'vpl%012x' % random.randrange(16**12)
    planetName = 'pri%012x' % random.randrange(16**12)
    starName = 'sec%012x' % random.randrange(16**12)
    sysFile = sysName + '.in'
    planetFile = planetName + '.in'
    starFile = starName + '.in'
    logfile = sysName + '.log'
    planetFwFile = '%s.planet.forward' % sysName
    starFwFile = '%s.star.forward' % sysName

    # Get the planet mass (all prior)
    dPlanetMass = -np.inf
    while -dPlanetMass > 10:
        inc = np.arccos(1 - np.random.random())
        msini = Mpsini + sigMpsini * np.random.randn()
        dPlanetMass = -msini / np.sin(inc)

    # Negative for vplanet
    dSatXUVTime *= -1
    dXUVBeta *= -1

    # Populate the planet input file (periods negative to make units days in VPLanet)
    planet_in = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", dMass1), planet_in)
    planet_in = re.sub("%s(.*?)#" % "dRotPeriod", "%s %.6e #" % ("dRotPeriod", -dProt1), planet_in)
    planet_in = re.sub("%s(.*?)#" % "dTidalTau", "%s %.6e #" % ("dTidalTau", dTau1), planet_in)

    with open(os.path.join(PATH, "output", planetFile), 'w') as f:
        print(planet_in, file = f)

    # Populate the star input file
    star_in = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", dMass2), star_in)
    star_in = re.sub("%s(.*?)#" % "dRotPeriod", "%s %.6e #" % ("dRotPeriod", -dProt2), star_in)
    star_in = re.sub("%s(.*?)#" % "dTidalTau", "%s %.6e #" % ("dTidalTau", dTau2), star_in)
    star_in = re.sub("%s(.*?)#" % "dOrbPeriod", "%s %.6e #" % ("dOrbPeriod", -dPorb), star_in)
    star_in = re.sub("%s(.*?)#" % "dEcc", "%s %.6e #" % ("dEcc", dEcc), star_in)
    with open(os.path.join(PATH, "output", starFile), 'w') as f:
        print(star_in, file = f)

    # Populate the system input file
    vpl_in = re.sub('%s(.*?)#' % "dStopTime", '%s %.6e #' % ("dStopTime", dStopTime), vpl_in)
    vpl_in = re.sub('%s(.*?)#' % "dOutputTime", '%s %.6e #' % ("dOutputTime", dOutputTime), vpl_in)
    vpl_in = re.sub('sSystemName(.*?)#', 'sSystemName %s #' % sysName, vpl_in)
    vpl_in = re.sub('saBodyFiles(.*?)#', 'saBodyFiles %s %s #' % (planetFile, starFile), vpl_in)
    with open(os.path.join(PATH, "output", sysFile), 'w') as f:
        print(vpl_in, file = f)

    # Run VPLANET and get the output, then delete the output files
    subprocess.call(["vplanet", sysFile], cwd = os.path.join(PATH, "output"))
    output = vpl.GetOutput(os.path.join(PATH, "output"), logfile = logfile)

    try:
        os.remove(os.path.join(PATH, "output", planetFile))
        os.remove(os.path.join(PATH, "output", starFile))
        os.remove(os.path.join(PATH, "output", sysFile))
        os.remove(os.path.join(PATH, "output", planetFwFile))
        os.remove(os.path.join(PATH, "output", starFwFile))
        os.remove(os.path.join(PATH, "output", logfile))
    except FileNotFoundError:
        # Run failed!
        return -np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Ensure we ran for as long as we set out to
    if not output.log.final.system.Age >= dStopTime:
        return -np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Get output parameters
    dRad1 = float(output.log.final.planet.Radius)
    dRad2 = float(output.log.final.star.Radius)
    dTeff1 = float(output.log.final.planet.Temperature)
    dTeff2 = float(output.log.final.star.Temperature)
    dLum1 = float(output.log.final.star.Luminosity)
    dLum2 = float(output.log.final.star.Luminosity)
    dPorb = float(output.log.final.star.OrbPeriod)
    dEcc = float(output.log.final.star.Eccentricity)
    dProt1 = float(output.log.final.planet.RotPer)
    dProt2 = float(output.log.final.star.RotPer)

    # Compute the likelihood using provided constraints, assuming we have
    # radius constraints for both stars
    lnlike = ((dRad1 - r1) / r1Sig) ** 2
    lnlike += ((dRad2 - r2) / r2Sig) ** 2
    if teff1 is not None:
        lnlike += ((dTeff1 - teff1) / teff1Sig) ** 2
    if teff2 is not None:
        lnlike += ((dTeff2 - teff2) / teff2Sig) ** 2
    if lum1 is not None:
        lnlike += ((dLum1 - lum1) / lum1Sig) ** 2
    if lum2 is not None:
        lnlike += ((dLum2 - lum2) / lum2Sig) ** 2
    if porb is not None:
        lnlike += ((dPorb - porb) / porbSig) ** 2
    if ecc is not None:
        lnlike += ((dEcc - ecc) / eccSig) ** 2
    if prot1 is not None:
        lnlike += ((dProt1 - prot1) / prot1Sig) ** 2
    if prot2 is not None:
        lnlike += ((dProt2 - prot2) / prot2Sig) ** 2
    lnlike = -0.5 * lnlike + lnprior

    # Return likelihood and blobs
    return lnlike, dProt1, dProt2, dPorb, dEcc, dRad1, dRad2, dLum1, dLum2, dTeff1, dTeff2
# end function


def GetEvol(x, **kwargs):
    """
    Run a VPLanet simulation for this initial condition vector, x
    """

    # Get the current vector
    dMass1, dMass2, dProt1, dProt2, dTau1, dTau2, dPorb, dEcc, dAge = x

    # Unlog tau, convect to yr
    dTau1 = (10 ** dTau1) / utils.YEARSEC
    dTau2 = (10 ** dTau2) / utils.YEARSEC

    # Convert age to yr from Gyr, set stop time, output time to age of system
    dStopTime = dAge * 1.0e9
    dOutputTime = dStopTime

    # Get the prior probability
    lnprior = kwargs["LnPrior"](x, **kwargs)

    # If not finite, invalid initial conditions
    if np.isinf(lnprior):
        return None

    # Get strings containing VPLanet input filex (they must be provided!)
    try:
        planet_in = kwargs.get("PLANETIN")
        star_in = kwargs.get("STARIN")
        vpl_in = kwargs.get("VPLIN")
    except KeyError as err:
        print("ERROR: must supply PLANETIN, STARIN, VPLIN.")
        raise

    # Randomize file names
    sysName = 'vpl%012x' % random.randrange(16**12)
    planetName = 'pri%012x' % random.randrange(16**12)
    starName = 'sec%012x' % random.randrange(16**12)
    sysFile = sysName + '.in'
    planetFile = planetName + '.in'
    starFile = starName + '.in'
    logfile = sysName + '.log'
    planetFwFile = '%s.planet.forward' % sysName
    starFwFile = '%s.star.forward' % sysName

    # Populate the planet input file (periods negative to make units days in VPLanet)
    planet_in = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", dMass1), planet_in)
    planet_in = re.sub("%s(.*?)#" % "dRotPeriod", "%s %.6e #" % ("dRotPeriod", -dProt1), planet_in)
    planet_in = re.sub("%s(.*?)#" % "dTidalTau", "%s %.6e #" % ("dTidalTau", dTau1), planet_in)
    with open(os.path.join(PATH, "output", planetFile), 'w') as f:
        print(planet_in, file = f)

    # Populate the star input file
    star_in = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", dMass2), star_in)
    star_in = re.sub("%s(.*?)#" % "dRotPeriod", "%s %.6e #" % ("dRotPeriod", -dProt2), star_in)
    star_in = re.sub("%s(.*?)#" % "dTidalTau", "%s %.6e #" % ("dTidalTau", dTau2), star_in)
    star_in = re.sub("%s(.*?)#" % "dOrbPeriod", "%s %.6e #" % ("dOrbPeriod", -dPorb), star_in)
    star_in = re.sub("%s(.*?)#" % "dEcc", "%s %.6e #" % ("dEcc", dEcc), star_in)
    with open(os.path.join(PATH, "output", starFile), 'w') as f:
        print(star_in, file = f)

    # Populate the system input file
    vpl_in = re.sub('%s(.*?)#' % "dStopTime", '%s %.6e #' % ("dStopTime", dStopTime), vpl_in)
    vpl_in = re.sub('%s(.*?)#' % "dOutputTime", '%s %.6e #' % ("dOutputTime", dOutputTime), vpl_in)
    vpl_in = re.sub('sSystemName(.*?)#', 'sSystemName %s #' % sysName, vpl_in)
    vpl_in = re.sub('saBodyFiles(.*?)#', 'saBodyFiles %s %s #' % (planetFile, starFile), vpl_in)
    with open(os.path.join(PATH, "output", sysFile), 'w') as f:
        print(vpl_in, file = f)

    # Run VPLANET and get the output, then delete the output files
    subprocess.call(["vplanet", sysFile], cwd = os.path.join(PATH, "output"))
    output = vpl.GetOutput(os.path.join(PATH, "output"), logfile = logfile)

    try:
        os.remove(os.path.join(PATH, "output", planetFile))
        os.remove(os.path.join(PATH, "output", starFile))
        os.remove(os.path.join(PATH, "output", sysFile))
        os.remove(os.path.join(PATH, "output", planetFwFile))
        os.remove(os.path.join(PATH, "output", starFwFile))
        os.remove(os.path.join(PATH, "output", logfile))
    except FileNotFoundError:
        # Run failed!
        return None

    # Ensure we ran for as long as we set out to
    if not output.log.final.system.Age >= dStopTime:
        return None

    # Return output
    return output
# end function


def RunMCMC(x0=None, ndim=9, nwalk=90, nsteps=1000, pool=None, backend=None,
            restart=False, **kwargs):
    """
    """

    # Ensure LnPrior, prior sample are in kwargs
    try:
        kwargs["PriorSample"]
    except KeyError as err:
        print("ERROR: Must supply PriorSample function!")
        raise
    try:
        kwargs["LnPrior"]
    except KeyError as err:
        print("ERROR: Must supply LnPrior function!")
        raise

    # Extract path
    PATH = kwargs["PATH"]

    print("Running MCMC...")

    # Get the input files, save them as strings
    with open(os.path.join(PATH, "planet.in"), 'r') as f:
        planet_in = f.read()
        kwargs["PLANETIN"] = planet_in
    with open(os.path.join(PATH, "star.in"), 'r') as f:
        star_in = f.read()
        kwargs["starIN"] = star_in
    with open(os.path.join(PATH, "vpl.in"), 'r') as f:
        vpl_in = f.read()
        kwargs["VPLIN"] = vpl_in

    # Set up backend to save results
    if backend is not None:
        # Set up the backend
        handler = emcee.backends.HDFBackend(backend)

        # If restarting from a previous interation, initialize backend
        if not restart:
            handler.reset(nwalk, ndim)

    # Populate initial conditions for walkers using random samples over prior
    if not restart:
        # If MCMC isn't initialized, just sample from the prior
        if x0 is None:
            x0 = np.array([kwargs["PriorSample"](**kwargs) for w in range(nwalk)])

    ### Run MCMC ###

    # Define blobs, blob data types
    dtype = [("dProt1F", np.float64), ("dProt2F", np.float64), ("dPorbF", np.float64),
             ("dEccF", np.float64), ("dRad1F", np.float64), ("dRad2F", np.float64),
             ("dLum1F", np.float64), ("dLum2F", np.float64), ("dTeff1F", np.float64),
             ("dTeff2F", np.float64)]

    # Initialize the sampler object
    sampler = emcee.EnsembleSampler(nwalk, ndim, LnLike, kwargs=kwargs, pool=pool,
                                    blobs_dtype=dtype, backend=handler)

    # Actually run the MCMC
    if restart:
        sampler.run_mcmc(None, nsteps)
    else:
        for ii, result in enumerate(sampler.sample(x0, iterations=nsteps)):
            print("MCMC: %d/%d..." % (ii + 1, nsteps))

    print("Done!")
# end function
