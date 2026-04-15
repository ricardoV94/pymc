#   Copyright 2026 - present The PyMC Developers
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

from pymc.distributions import HalfNormal, Normal
from pymc.initial_point import make_initial_point_fn
from pymc.model.core import Model
from pymc.model.transform.reparam import automatic_reparametrization


def test_normal_reparam():
    coords = {"group": [0, 1, 2], "week": range(30)}
    with Model(coords=coords) as m:
        pop_mu = Normal("pop_mu", 0, 1)
        pop_sigma = HalfNormal("pop_sigma", 1)
        group_mu = Normal("ind_mu", pop_mu, pop_sigma, dims="group")
        obs = Normal(
            "obs", group_mu[np.tile([0, 1, 2], reps=10)], observed=np.zeros(30), dims="week"
        )

    reparam_m, params = automatic_reparametrization(m)

    assert len(params) == 1
    (param,) = params
    assert param.name == "ind_mu_lam_logit"
    assert param.get_value(borrow=True).shape == ()

    assert param in reparam_m.data_vars
    assert reparam_m["ind_mu"] in reparam_m.deterministics
    assert reparam_m["pop_mu"] in reparam_m.free_RVs
    assert reparam_m["pop_sigma"] in reparam_m.free_RVs

    # Original dims are propagated onto the reparam RV and the deterministic.
    dims = reparam_m.named_vars_to_dims
    assert dims["ind_mu"] == ["group"]
    assert dims["ind_mu_reparam"] == ["group"]
    # PyMC doesn't store dims for scalar parameters?
    assert dims.get("ind_mu_lam_logit", ()) == ()

    # Drive lam to 1 (fully centered) — the reparam model must then reproduce
    # the original logp exactly when evaluated at matching points.
    reparam_test_vals = make_initial_point_fn(model=reparam_m, default_strategy="prior")(seed=36)
    reparam_logp = reparam_m.compile_logp()

    test_vals = reparam_test_vals.copy()
    test_vals["ind_mu"] = test_vals.pop("ind_mu_reparam")
    orig_logp = m.compile_logp()(test_vals)

    param.set_value(np.asarray(20.0))  # sigmoid(20) ~= 1
    new_logp = reparam_logp(reparam_test_vals)
    np.testing.assert_allclose(new_logp, orig_logp, rtol=1e-6)

    param.set_value(np.asarray(0.0))
    assert not np.allclose(new_logp, reparam_logp(reparam_test_vals))


def test_partial_broadcast_reparam():
    coords = {"group": range(3), "rep": range(10)}
    with Model(coords=coords) as m:
        pop_mu = Normal("pop_mu", 0, 1, dims=("group"))
        x = Normal("x", pop_mu[:, None], 1.0, dims=("group", "rep"))

    reparam_m, params = automatic_reparametrization(m)

    # pop_mu is a root RV → not rewritten.
    assert reparam_m["pop_mu"] in reparam_m.free_RVs
    assert "pop_mu_reparam" not in reparam_m.named_vars

    assert len(params) == 1
    (param,) = params
    assert param.name == "x_lam_logit"
    # Only the ``group`` axis survives; ``rep`` collapsed because both loc and
    # scale are broadcast along it.
    assert param.get_value(borrow=True).shape == (3,)

    # Original dims are propagated; the lam param keeps only the surviving axis.
    dims = reparam_m.named_vars_to_dims
    assert dims["x"] == ["group", "rep"]
    assert dims["x_reparam"] == ["group", "rep"]
    assert dims["x_lam_logit"] == ["group"]

    # Reparam model still evaluates to a finite logp at the initial point.
    assert np.isfinite(reparam_m.compile_logp()(reparam_m.initial_point()))


def test_gradient_function():
    """ "Test we can build something like.

    def build_reparametrized_logp(logp):
        \"""
        Returns:
          - transformed_logp(φ, ψ) → log p(T_ψ(φ)) + log|det J_T(φ)|
          - grad_φ(φ, ψ) → gradient w.r.t. φ (for the sampler / NUTS)
          - grad_ψ(φ, ψ) → gradient w.r.t. ψ (for optimizing the flow) (given the gradient wrt to the parameters already computed)
        \"""
    """
    assert 0
