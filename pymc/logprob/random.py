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
from pytensor.tensor import exp
from pytensor.tensor.random import lognormal, normal

from pymc.logprob.rewriting import (
    MeasurablePatternNodeRewriter,
    random_variables_rewrites_db,
)

rv_args = ("rng", "dtype", "size")


exp_normal_to_lognormal = MeasurablePatternNodeRewriter(
    (exp, (normal, *rv_args, "mu", "sigma")),
    (lognormal, *rv_args, "mu", "sigma"),
)


random_variables_rewrites_db.register(
    "exp_normal_to_lognormal",
    exp_normal_to_lognormal,
    "basic",
    "random",
)
