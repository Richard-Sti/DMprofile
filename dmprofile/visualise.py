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

"""
import sys
import numpy
import csv
import sympy

sys.path.append("/mnt/zfsusers/rstiskalek/DMprofile/ESR")  # noqa
from esr.fitting.sympy_symbols import (a0, a1, a2, a3, x)  # noqa  # type: ignore
from esr.generation import simplifier # noqa  # type: ignore


def read_functions(comp, likelihood, nfunc=50, vmin=1e-50):
    """
    Read the final likelihood file and convert the function strint into a
    Python function.

    Parameters
    ----------
    comp : int
        Complexity.
    likelihood : likelihood object
        Likelihood object.
    nfunc : int, optional
        Maximum number of functions to extract.

    Returns
    -------
    funcs : list of functions
        ESR fitted functions.
    measured : array of 1-dimensional arrays
        ESR functions' hyperapaameters.
    DL : 1-dimensional array
        Description lengths.
    fcn_list : list of strings
        ESR functions' string representation.
    """
    max_param = 4  # ASK

    # We read the final likelihood file and extract its function string
    # representation, their paramaters and the description length
    with open(likelihood.out_dir + '/final_'+str(comp)+'.dat', "r") as f:
        reader = csv.reader(f, delimiter=';')
        data = [row for row in reader]

    fcn_list = [d[1] for d in data]
    params = numpy.array([d[-4:] for d in data], dtype=float)  # ASK

    DL = numpy.array([d[2] for d in data], dtype=float)

    # We now select only good number of functions.
    alpha = numpy.exp(numpy.amin(DL[numpy.isfinite(DL)]) - DL)  # ASK
    m = alpha > 1e-50
    nfunc = min(nfunc, numpy.sum(m))
    fcn_list = [d for i, d in enumerate(fcn_list) if m[i]]
    params = params[m, :][:nfunc, :]
    alpha = alpha[m][:nfunc]
    DL = DL[m][:nfunc]

    # We now convert the function strings into Python functions
    funcs = [None] * nfunc
    measured = [None] * nfunc
    for i in range(nfunc):
        fcn_i = fcn_list[i].replace('\'', '')
        k = simplifier.count_params([fcn_i], max_param)[0]

        variables = [x, a0, a1, a2, a3][:k + 1]
        fcn_i, eq, __ = likelihood.run_sympify(fcn_i)
        funcs[i] = sympy.lambdify(variables, eq, modules=["numpy"])
        measured[i] = params[i, :k]

    measured_params = numpy.asanyarray(measured, dtype=object)
    return funcs, measured_params, DL, fcn_list
