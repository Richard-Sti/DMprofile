# Copyright (C) 2022 Richard Stiskalek
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

from scipy.integrate import quad
from scipy.stats import binned_statistic
from scipy.special import factorial
import numpy
import sympy

from tqdm import tqdm

# from esr.generation import simplifier  # noqa
# from esr import generation
import sys
sys.path.append("/mnt/zfsusers/rstiskalek/DMprofile/ESR")

from esr.generation import simplifier  # noqa
# from esr.fitting.sympy_symbols import *
from esr.fitting.sympy_symbols import (inv, square, cube, sqrt, log, pow, x,
                                       a0, a1, a2)


class Likelihood:
    """
    Likelihood class used to fit DM density profiles on N-body simulations with
    Exhaustive Symbolic Regression.
    """
    def __init__(self, datapath, mpart=1.1641532e-10):
        esr_dir = os.path.abspath(os.path.join(os.path.dirname(simplifier.__file__), '..', '')) + '/'  # noqa
        self.data_dir = esr_dir + '/data/'
        self.data_file = self.data_dir + '/CC_Hubble.dat'
        self.fn_dir = esr_dir + "function_library/core_maths/"
        self.like_dir = esr_dir + "/fitting/"
        self.like_file = "likelihood_cc"
        self.sym_file = "symbols_cc"
        self.fnprior_prefix = "aifeyn_"
        self.combineDL_prefix = "combine_DL_"
        self.final_prefix = "final_"

        self.base_out_dir = self.like_dir + "/output/"
        self.temp_dir = self.base_out_dir + "/partial_cc_dimful"
        self.out_dir = self.base_out_dir + "/output_cc_dimful"
        self.fig_dir = self.base_out_dir + "/figs_cc_dimful"

        self.xlabel = r"$r$"
        self.ylabel = r"$\rho(r)$"

        archive = numpy.load(datapath)

        data = archive["6"]
        print(data.size)
        rmin, rmax = numpy.min(data["r"]), numpy.max(data["r"])
        bins = numpy.linspace(rmin, rmax, 1000)
        self.mpart = mpart
        self.xvar = (bins[1:] + bins[:-1]) / 2
        self.yvar, __, __ = binned_statistic(data["r"], data["M"] / mpart,
                                             statistic="sum", bins=bins,
                                             range=(rmin, rmax))
        self.dx = bins[1] - bins[0]
        self.yerr = None

    def get_pred(self, r, a, eq_numpy, **kwargs):
        """
        Calculate the predicted DM density profile at radial locations `r`.
        TODO: edit docs

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

    def clear_data(self):
        """
        Clear data, used for numerical integration. However this is not needed
        in the current implementation.
        """
        pass

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
        expnum = self.get_pred(self.xvar, numpy.atleast_1d(a), eq_numpy)

        negll = numpy.sum(expnum - self.yvar * numpy.log(expnum))

        if numpy.isnan(negll):
            return numpy.infty
        return negll

    def run_sympify(self, fcn_i, **kwargs):
        """
        Sympify a function

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

        eq = sympy.sympify(
            fcn_i, locals={"inv": inv,
                           "square": square,
                           "cube": cube,
                           "sqrt": sqrt,
                           "log": log,
                           "pow": pow,
                           "x": x,
                           "a0": a0,
                           "a1": a1,
                           "a2": a2})
        return fcn_i, eq, False
