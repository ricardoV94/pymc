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
import pytest

from pytensor import tensor as pt
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    IncSubtensor,
)
from scipy.stats import distributions as sp

from pymc import logp
from pymc.logprob import factorized_joint_logprob


@pytest.mark.parametrize(
    "indices, size",
    [
        (slice(0, 2), 5),
        (np.r_[True, True, False, False, True], 5),
        (np.r_[0, 1, 4], 5),
        ((np.array([0, 1, 4]), np.array([0, 1, 4])), (5, 5)),
    ],
)
def test_measurable_incsubtensor(indices, size):
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

    Y_rv_logp = factorized_joint_logprob({Y_rv: y_value_var})
    Y_rv_logp_combined = pt.add(*Y_rv_logp.values())

    obs_logps = Y_rv_logp_combined.eval({y_value_var: y_val})

    y_val_idx = y_val.copy()
    y_val_idx[indices] = data
    exp_obs_logps = sp.norm.logpdf(y_val_idx, mu, sigma)

    np.testing.assert_almost_equal(obs_logps, exp_obs_logps)


def test_subtensor():
    rv_shape = (5, 3, 2)
    mu = np.arange(np.prod(rv_shape)).reshape(rv_shape)
    x = pt.random.normal(mu, size=rv_shape)
    y = pt.vector("y")
    z = pt.scalar("z")
    indices = pt.tensor3("mask", dtype=bool)

    underlying_rv = pt.exp(pt.add(x, y, z))
    subtensor_rv = underlying_rv[indices]
    subtensor_vv = subtensor_rv.clone()
    subtensor_logp = logp(subtensor_rv, subtensor_vv)

    test_y = np.linspace(1.7, 2.5, rv_shape[-1])
    test_z = 0.3
    test_indices = np.random.binomial(n=1, p=0.5, size=rv_shape).astype(bool)
    test_underlying_vv = np.zeros(rv_shape)
    test_subtensor_vv = test_underlying_vv[test_indices]
    np.testing.assert_almost_equal(
        subtensor_logp.eval({y: test_y, z: test_z, subtensor_vv: test_subtensor_vv, indices: test_indices}),
        logp(underlying_rv, test_underlying_vv).eval({y: test_y, z: test_z})[test_indices],
    )