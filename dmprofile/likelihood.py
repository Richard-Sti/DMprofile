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

import numpy
import sympy
from scipy.stats import binned_statistic

sys.path.append("/mnt/zfsusers/rstiskalek/DMprofile/ESR")
from esr.fitting.sympy_symbols import (a0, a1, a2, cube, inv, log, pow, sqrt,
                                       square, x)
from esr.generation import simplifier


class PoissonLikelihood:
    """
    Poisson likelihood class to fit DM density profiles on N-body simulations.
    Calculated on binned data such that the predicted number of particles in a
    bin is assumed to be Poisson distributed.

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

    def __init__(self, datapath, name, mpart=1.1641532e-10):
        # First of all, we set up the ESR paths
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

        # These are used in ESR diagnostics plots.
        self.xlabel = r"$r$"
        self.ylabel = r"$\rho(r)$"

        # Now we load the data. TODO: later do this directly when generating
        # the data to avoid unnecessary overhead here.
        archive = numpy.load(datapath)
        data = archive["42"]  # Good test halo for now.
        rmin, rmax = numpy.min(data["r"]), numpy.max(data["r"])
        bins = 10**numpy.arange(numpy.log10(rmin), numpy.log10(rmax), 0.005)
        self.mpart = mpart
        self.xvar = (bins[1:] + bins[:-1]) / 2
        self.yvar, __, __ = binned_statistic(data["r"], data["M"] / mpart,
                                             statistic="sum", bins=bins,
                                             range=(rmin, rmax))
        self.dx = numpy.diff(bins)
        self.yerr = None

    def get_pred(self, r, a, eq_numpy, **kwargs):
        """
        Calculated the predicted number of particles in a bin centered at
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
        rho : 1-dimensional array
            The predicted DM density profile at radial locations `r`.
        """
        return 4 * numpy.pi * r**2 * eq_numpy(r, *a) / self.mpart * self.dx

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

    def run_sympify(self, fcn_i, **kwargs):
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
