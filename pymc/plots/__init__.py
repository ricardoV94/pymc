#   Copyright 2023 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Alias for the `plots` submodule from ArviZ.

Plots are delegated to the ArviZ library, a general purpose library for
"exploratory analysis of Bayesian models."
See https://arviz-devs.github.io/arviz/ for details on plots.
"""
import functools
import sys

import arviz as az

# Makes this module as identical to arviz.plots as possible
for attr in az.plots.__all__:
    obj = getattr(az.plots, attr)
    if not attr.startswith("__"):
        setattr(sys.modules[__name__], attr, obj)

DEPRECATED_PLOTS_ALIASES = dict(
    autocorrplot="plot_autocorr",
    forestplot="plot_forest",
    kdeplot="plot_kde",
    energyplot="plot_energy",
    densityplot="plot_density",
    pairplot="plot_pair",
    traceplot="plot_trace",
    compareplot="plot_compare",
)


def __getattr__(name):
    if name in DEPRECATED_PLOTS_ALIASES:
        new_name = DEPRECATED_PLOTS_ALIASES[name]
        raise ImportError(
            f"The function `{name}` from PyMC was an alias for `{new_name}` from "
            f"ArviZ. It was removed in PyMC 4.0. "
            f"Switch to `pymc.{new_name}` or `arviz.{new_name}`."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
