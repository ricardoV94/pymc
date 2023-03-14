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
#
#   MIT License
#
#   Copyright (c) 2021-2022 aesara-devs
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.stats.distributions as sp

from pytensor.graph.basic import equal_computations
from pytensor.graph.rewriting.basic import in2out
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    IncSubtensor,
    Subtensor,
)

from pymc.logprob.rewriting import (
    MeasurablePatternNodeRewriter,
    PreserveRVMappings,
    local_lift_DiracDelta,
)
from pymc.logprob.utils import DiracDelta, dirac_delta
from tests.logprob.utils import joint_logprob


def test_local_lift_DiracDelta():
    c_at = pt.vector()
    dd_at = dirac_delta(c_at)

    Z_at = pt.cast(dd_at, "int64")

    res = rewrite_graph(Z_at, custom_rewrite=in2out(local_lift_DiracDelta), clone=False)
    assert isinstance(res.owner.op, DiracDelta)
    assert isinstance(res.owner.inputs[0].owner.op, Elemwise)

    Z_at = dd_at.dimshuffle("x", 0)

    res = rewrite_graph(Z_at, custom_rewrite=in2out(local_lift_DiracDelta), clone=False)
    assert isinstance(res.owner.op, DiracDelta)
    assert isinstance(res.owner.inputs[0].owner.op, DimShuffle)

    Z_at = dd_at[0]

    res = rewrite_graph(Z_at, custom_rewrite=in2out(local_lift_DiracDelta), clone=False)
    assert isinstance(res.owner.op, DiracDelta)
    assert isinstance(res.owner.inputs[0].owner.op, Subtensor)

    # Don't lift multi-output `Op`s
    c_at = pt.matrix()
    dd_at = dirac_delta(c_at)
    Z_at = pt.nlinalg.svd(dd_at)[0]

    res = rewrite_graph(Z_at, custom_rewrite=in2out(local_lift_DiracDelta), clone=False)
    assert res is Z_at


def test_local_remove_DiracDelta():
    c_at = pt.vector()
    dd_at = dirac_delta(c_at)

    fn = pytensor.function([c_at], dd_at)
    assert not any(isinstance(node.op, DiracDelta) for node in fn.maker.fgraph.toposort())


@pytest.mark.parametrize(
    "indices, size",
    [
        (slice(0, 2), 5),
        (np.r_[True, True, False, False, True], 5),
        (np.r_[0, 1, 4], 5),
        ((np.array([0, 1, 4]), np.array([0, 1, 4])), (5, 5)),
    ],
)
def test_joint_logprob_incsubtensor(indices, size):
    """Make sure we can compute a joint log-probability for ``Y[idx] = data`` where ``Y`` is univariate."""

    rng = np.random.RandomState(232)
    mu = np.power(10, np.arange(np.prod(size))).reshape(size)
    sigma = 0.001
    data = rng.normal(mu[indices], 1.0)
    y_val = rng.normal(mu, sigma, size=size)

    Y_base_rv = pt.random.normal(mu, sigma, size=size)
    Y_rv = pt.set_subtensor(Y_base_rv[indices], data)
    Y_rv.name = "Y"
    y_value_var = Y_rv.clone()
    y_value_var.name = "y"

    assert isinstance(Y_rv.owner.op, (IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1))

    Y_rv_logp = joint_logprob({Y_rv: y_value_var}, sum=False)

    obs_logps = Y_rv_logp.eval({y_value_var: y_val})

    y_val_idx = y_val.copy()
    y_val_idx[indices] = data
    exp_obs_logps = sp.norm.logpdf(y_val_idx, mu, sigma)

    np.testing.assert_almost_equal(obs_logps, exp_obs_logps)


rv_args = ("rng", "dtype", "size")


class TestMeasurablePatternNodeRewriter:
    def test_basic(self):
        test_pattern = MeasurablePatternNodeRewriter(
            (pt.exp, (pt.log, "x")),
            (pt.cos, "x"),
        )

        zero = pt.constant(0.0)
        x = pt.exp(pt.log(zero))

        fgraph = pytensor.graph.FunctionGraph(outputs=[x])
        [res] = test_pattern.transform(fgraph, fgraph.outputs[0].owner)
        expected_res = pt.cos(zero)
        assert equal_computations([res], [expected_res])

    def test_random(self):
        test_pattern = MeasurablePatternNodeRewriter(
            (pt.exp, (pt.random.normal, *rv_args, (pt.abs, "mu"), "sigma")),
            (pt.random.lognormal, *rv_args, (pt.abs, (pt.neg, "mu")), "sigma"),
        )

        rng = pytensor.shared(np.random.default_rng())
        mu, sigma = pt.scalars("mu", "sigma")

        x = pt.exp(pt.random.normal(mu, sigma, rng=rng))
        fgraph = pytensor.graph.FunctionGraph(outputs=[x], clone=False)
        res = test_pattern.transform(fgraph, fgraph.outputs[0].owner)
        assert res is False

        x = pt.exp(pt.random.normal(pt.abs(mu), sigma, rng=rng))
        fgraph = pytensor.graph.FunctionGraph(outputs=[x], clone=False)
        [res] = test_pattern.transform(fgraph, fgraph.outputs[0].owner)
        expected_res = pt.random.lognormal(pt.abs(pt.neg(mu)), sigma, rng=rng)
        assert equal_computations([res], [expected_res])

    def test_intermediate_value_variable_protected(self):
        test_pattern = MeasurablePatternNodeRewriter(
            (pt.exp, (pt.random.normal, *rv_args, "mu", "sigma")),
            (pt.random.lognormal, *rv_args, "mu", "sigma"),
        )

        x = pt.random.normal()
        y = pt.random.normal(x)
        z = pt.exp(y)

        fgraph = pytensor.graph.FunctionGraph(
            outputs=[x, z],
            clone=False,
            # z is just a deterministic transformation of y, we don't want to rewrite it!
            features=[PreserveRVMappings({x: x.type(), y: y.type()})],
        )
        res = test_pattern.transform(fgraph, fgraph.outputs[-1].owner)
        assert res is False

        fgraph = pytensor.graph.FunctionGraph(
            outputs=[x, z],
            clone=False,
            # z is now a measurable transformation, because y is not valued
            features=[PreserveRVMappings({x: x.type(), z: z.type()})],
        )
        [res] = test_pattern.transform(fgraph, fgraph.outputs[-1].owner)
        assert res.owner.op == pt.random.lognormal

    def test_intermediate_vars_multiple_clients(self):
        test_pattern = MeasurablePatternNodeRewriter(
            (pt.exp, (pt.random.normal, *rv_args, "mu", "sigma")),
            (pt.random.lognormal, *rv_args, "mu", "sigma"),
        )

        x = pt.random.normal()
        x_cl = pt.random.normal(x)
        y = pt.random.normal(x)
        y_cl = pt.random.normal(y)
        y_target = pt.exp(y)

        fgraph = pytensor.graph.FunctionGraph(
            # y has two clients, so rewrite can't be applied
            outputs=[x, x_cl, y, y_cl, y_target],
            clone=False,
        )
        res = test_pattern.transform(fgraph, fgraph.outputs[-1].owner)
        assert res is False

        fgraph = pytensor.graph.FunctionGraph(
            # Now it's fine
            outputs=[x, x_cl, y, y_target],
            clone=False,
        )
        [res] = test_pattern.transform(fgraph, fgraph.outputs[-1].owner)
        assert res.owner.op == pt.random.lognormal
