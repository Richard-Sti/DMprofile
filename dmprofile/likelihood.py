# Copyright (C) 2023 Richard Stiskalek, Deaglan Bartlett
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
ESR likelihood for fitting DM density profiles on N-body simulations.
"""
import os
import sys
from abc import ABC, abstractmethod

import numpy
import sympy
from scipy.integrate import quad
from scipy.stats import binned_statistic

sys.path.append("/mnt/zfsusers/rstiskalek/DMprofile/ESR")  # noqa
from esr.fitting.sympy_symbols import (a0, a1, a2, cube, inv, log, pow, sqrt, square, x) # noqa  # type: ignore
from esr.generation import simplifier  # noqa  # type: ignore


###############################################################################
#                          Likelihood classes                                 #
###############################################################################


class BaseLikelihood(ABC):
    """
    Base class for ESR likelihoods.
    """
    xlabel = None
    ylabel = None

    def set_paths(self, name):
        """Setup paths for ESR."""
        esr_dir = os.path.abspath(os.path.join(os.path.dirname(simplifier.__file__), '..', '')) + '/'  # noqa
        self.fn_dir = esr_dir + "function_library/core_maths/"
        self.like_dir = esr_dir + "/fitting/"
        self.like_file = f"likelihood_{name}"
        self.sym_file = f"symbols_{name}"
        self.fnprior_prefix = "aifeyn_"
        self.combineDL_prefix = "combine_DL_"
        self.final_prefix = "final_"

        self.base_out_dir = self.like_dir + "/output/"
        self.temp_dir = self.base_out_dir + f"/partial_{name}_dimful"
        self.out_dir = self.base_out_dir + f"/output_{name}_dimful"
        self.fig_dir = self.base_out_dir + f"/figs_{name}_dimful"

    @staticmethod
    def counts_in_bins(radpos, mass, dlogr, mpart):
        """
        Count particles in bins of width `dlogr` in log-space. In case of
        unequal masses, a single occurence is counted as mass in units of
        `mpart`.

        Parameters
        ----------
        radpos : 1-dimensional array
            Radial positions of particles.
        mass : 1-dimensional array
            Masses of particles.
        dlogr : float
            Width of bins in log-space.
        mpart : float
            Mass of a single DM particle in the simulation.

        Returns
        -------
        bin_centres : 1-dimensional array
            Bin centres.
        counts : 1-dimensional array
            Number of particles in each bin.
        bin_widths : 1-dimensional array
            Bin width.
        """
        rmin, rmax = numpy.min(radpos), numpy.max(radpos)
        bins = 10**numpy.arange(numpy.log10(rmin), numpy.log10(rmax), dlogr)
        counts, __, __ = binned_statistic(
            radpos, mass / mpart, statistic="sum", bins=bins,
            range=(rmin, rmax))
        bin_widths = numpy.diff(bins)
        bin_centres = (bins[1:] + bins[:-1]) / 2
        return bin_centres, counts, bin_widths

    @staticmethod
    def run_sympify(fcn_i, **kwargs):
        """
        Sympify a function.

        Parameters
        ----------
        fcn_i : str
            String representing function we wish to fit to data.
        **kwargs : dict
            Additional keyword arguments. Currently not supported.

        Returns
        -------
        fcn_i : str
            String representing function we wish to fit to data with
            superfluous characters removed.
        eq : sympy object
            Sympy object representing function we wish to fit to data.
        integrated : bool
            Whether we analytically integrated the function or not.
        """
        fcn_i = fcn_i.replace('\n', '')
        fcn_i = fcn_i.replace('\'', '')
        locs = {"inv": inv, "square": square, "cube": cube,
                "sqrt": sqrt, "log": log, "pow": pow, "x": x, "a0": a0,
                "a1": a1, "a2": a2
                }
        eq = sympy.sympify(fcn_i, locals=locs)
        return fcn_i, eq, False

    @abstractmethod
    def get_pred(self, r, a, eq_numpy, **kwargs):
        pass

    @abstractmethod
    def get_pred_bins(self, dlogr, a, eq_numpy, **kwargs):
        pass

    @abstractmethod
    def negloglike(self, a, eq_numpy, **kwargs):
        pass


class PoissonLikelihood(BaseLikelihood):
    """
    Poisson likelihood class to fit DM density profiles. Calculated on binned
    data such that the predicted number of particles in a bin is assumed to be
    Poisson distributed.

    Parameters
    ----------
    datapath : str
        Path to the data file containing the DM density profile.
    name : str
        Name of this run.
    mpart : float, optional
        Mass of a single DM particle in the simulation. Default is
        1.1641532e-10.
    """
    yerr = None

    def __init__(self, datapath, name, dlogr=0.005, mpart=1.1641532e-10):
        # First of all, we set up the ESR paths
        self.set_paths("poisson_" + name)

        # Now we load the data. TODO: later do this directly when generating
        # the data to avoid unnecessary overhead here.
        self.archive = numpy.load(datapath)
        data = self.archive["42"]  # Good test halo for now.
        self.mpart = mpart

        self.xvar, self.yvar, self.dx = self.counts_in_bins(
            data['r'], data['M'], dlogr=dlogr, mpart=mpart)

    def get_pred(self, r, a, eq_numpy, **kwargs):
        """
        Calculate the predicted number of particles in a bin centered at
        radial location `r`.

        Parameters
        ----------
        r : 1-dimensional array
            Radial locations at which to evaluate the DM density profile.
        a : list
            Parameters to substitute into the equation considered.
        eq_numpy : numpy function
            Function to use which gives DM density profile.
        **kwargs : dict
            Additional keyword arguments. Currently not supported.

        Returns
        -------
        npart : 1-dimensional array
            Expected number of particles in a bin.
        """
        return 4 * numpy.pi * r**2 * eq_numpy(r, *a) / self.mpart * self.dx

    def get_pred_bins(self, dlogr, a, eq_numpy, **kwargs):
        data = self.archive["42"]

        bin_centres, counts, dx = self.counts_in_bins(
            data["r"], data["M"], dlogr=dlogr, mpart=self.mpart)

        pred_counts = 4 * numpy.pi * bin_centres**2 * eq_numpy(bin_centres, *a)
        pred_counts *= dx / self.mpart

        return bin_centres, pred_counts, counts

    def negloglike(self, a, eq_numpy, **kwargs):
        """
        Calculate the negative log-likelihood for a given function.

        Parameters
        ----------
        a : list
            Parameters to substitute into the equation considered.
        eq_numpy : numpy function
            Function to use which gives DM density profile.
        **kwargs : dict
            Additional keyword arguments. Currently not supported.

        Returns
        -------
        nll : float
            The negative log-likelihood for the given function and parameters.
        """
        # We first calculate the expected count in each bin, then the negative
        # log likelihood. We ignore the normalisation term.
        expnum = self.get_pred(self.xvar, numpy.atleast_1d(a), eq_numpy)
        negll = numpy.sum(expnum - self.yvar * numpy.log(expnum))
        if not numpy.isfinite(negll):
            return numpy.infty
        return negll


class ParticleLikelihood(BaseLikelihood):
    """
    Particle based likelihood class to fit DM density profiles.

    Parameters
    ----------
    datapath : str
        Path to the data file containing the DM density profile.
    name : str
        Name of this run.
    mpart : float, optional
        Mass of a single DM particle in the simulation. Default is
        1.1641532e-10.
    """
    yvar = None
    yerr = None

    def __init__(self, datapath, name, mpart=1.1641532e-10):
        # First of all, we set up the ESR paths
        self.set_paths("particle" + name)

        # Now we load the data. TODO: later do this directly when generating
        # the data to avoid unnecessary overhead here.
        archive = numpy.load(datapath)
        data = archive["42"]  # Good test halo for now.
        self.mpart = mpart
        self.bounds = (numpy.min(data["r"]), numpy.max(data["r"]))
        self.xvar = data["r"]
        self.mass = data["M"]
        self.mtot = numpy.sum(data["M"])

    def get_pred(self, r, a, eq_numpy, **kwargs):
        """
        Calculate the predicted number density of particles at a radial
        location `r`.

        Parameters
        ----------
        r : 1-dimensional array
            Radial locations at which to evaluate the DM density profile.
        a : list
            Parameters to substitute into the equation considered.
        eq_numpy : numpy function
            Function to use which gives DM density profile.
        **kwargs : dict
            Additional keyword arguments. Currently not supported.

        Returns
        -------
        rho : 1-dimensional array
            Number density of particles at radial locations `r`.
        """
        return 4 * numpy.pi * r**2 * eq_numpy(r, *a)

    def get_pred_bins(self, dlogr, a, eq_numpy, **kwargs):
        bin_centres, counts, dx = self.counts_in_bins(
            self.xvar, self.mass, dlogr=dlogr, mpart=self.mpart)

        pred_counts = self.get_pred(bin_centres, a, eq_numpy, **kwargs)
        pred_counts *= dx / self.mpart
        return bin_centres, pred_counts, counts

    def negloglike(self, a, eq_numpy, **kwargs):
        """
        Calculate the negative log-likelihood for a given function.

        Parameters
        ----------
        a : list
            Parameters to substitute into the equation considered.
        eq_numpy : numpy function
            Function to use which gives DM density profile.
        **kwargs : dict
            Additional keyword arguments. Currently not supported.

        Returns
        -------
        nll : float
            The negative log-likelihood for the given function and parameters.
        """
        args = (numpy.atleast_1d(a), eq_numpy,)
        # TODO do something about the convergence.
        mtot = quad(self.get_pred, *self.bounds, args=args)[0]
        negll = -numpy.sum(numpy.log(self.get_pred(self.xvar, *args) / mtot))

        if not numpy.isfinite(negll):
            return numpy.infty
        return negll
