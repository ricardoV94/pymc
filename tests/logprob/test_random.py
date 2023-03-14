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
import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.graph.basic import equal_computations

import pymc as pm

from pymc.logprob.rewriting import construct_ir_fgraph


def test_exp_normal_to_lognormal():
    rng = pytensor.shared(np.random.default_rng(), name="rng").clone()
    size = pt.vector("size", dtype="int64", shape=(2,))

    mu, sigma = pt.scalars("mu", "sigma")
    x = pt.exp(pm.Normal.dist(mu, sigma, rng=rng, size=size))

    fgraph, *_ = construct_ir_fgraph({x: x.type})
    res = fgraph.outputs
    expected_res = [pm.LogNormal.dist(mu, sigma, rng=rng, size=size)]
    assert equal_computations(res, expected_res)
