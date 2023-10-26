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

# pylint: disable=wildcard-import

import logging

_log = logging.getLogger(__name__)

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)


def __set_compiler_flags():
    # Workarounds for PyTensor compiler problems on various platforms
    import pytensor

    current = pytensor.config.gcc__cxxflags
    augmented = f"{current} -Wno-c++11-narrowing"

    # Work around compiler bug in GCC < 8.4 related to structured exception
    # handling registers on Windows.
    # See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=65782 for details.
    # First disable C++ exception handling altogether since it's not needed
    # for the C extensions that we generate.
    augmented = f"{augmented} -fno-exceptions"
    # Now disable the generation of stack unwinding tables.
    augmented = f"{augmented} -fno-unwind-tables -fno-asynchronous-unwind-tables"

    pytensor.config.gcc__cxxflags = augmented


__set_compiler_flags()

from pymc import _version, gp, ode, plots, sampling, stats
from pymc.data import *
from pymc.distributions import *
from pymc.func_utils import find_constrained_prior
from pymc.logprob import *
from pymc.model.core import *
from pymc.model.transform.conditioning import do, observe
from pymc.model_graph import model_to_graphviz, model_to_networkx
from pymc.plots import DEPRECATED_PLOTS_ALIASES
from pymc.pytensorf import *
from pymc.sampling import *
from pymc.smc import *
from pymc.step_methods import *
from pymc.tuning import *
from pymc.variational import *

__version__ = _version.get_versions()["version"]


def __getattr__(name):
    if name in DEPRECATED_PLOTS_ALIASES:
        new_name = DEPRECATED_PLOTS_ALIASES[name]
        raise ImportError(
            f"The function `{name}` from PyMC was an alias for `{new_name}` from "
            f"ArviZ. It was removed in PyMC 4.0. "
            f"Switch to `pymc.{new_name}` or `arviz.{new_name}`."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
