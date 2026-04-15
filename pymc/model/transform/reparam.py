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
import sys

from itertools import zip_longest

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.graph.rewriting.basic import dfs_rewriter, node_rewriter
from pytensor.graph.rewriting.db import RewriteDatabaseQuery, SequenceDB
from pytensor.graph.traversal import ancestors
from pytensor.tensor.random.basic import (
    CauchyRV,
    GumbelRV,
    LaplaceRV,
    LogisticRV,
    NormalRV,
    StudentTRV,
)
from pytensor.tensor.sharedvar import TensorSharedVariable
from pytensor.tensor.type_other import NoneTypeT

from pymc.model.core import Model
from pymc.model.fgraph import (
    ModelFreeRV,
    fgraph_from_model,
    model_deterministic,
    model_free_rv,
    model_from_fgraph,
    model_named,
)
from pymc.util import get_transformed_name

# RV Op type -> (loc_index, scale_index) into `op.dist_params(node)`
LOC_SCALE_FAMILIES: dict[type, tuple[int, int]] = {
    NormalRV: (0, 1),
    CauchyRV: (0, 1),
    LaplaceRV: (0, 1),
    LogisticRV: (0, 1),
    GumbelRV: (0, 1),
    StudentTRV: (1, 2),
}


def _is_broadcasted(ref, *args) -> bool:
    ref_bcast = ref.type.broadcastable
    for arg in args:
        pairs = reversed(tuple(zip_longest(arg.type.broadcastable, ref_bcast, fillvalue=True)))
        if any(a_bc and not r_bc for a_bc, r_bc in pairs):
            return True
    return False


def _depends_on_free_rv(vars) -> bool:
    return any(
        anc.owner is not None and isinstance(anc.owner.op, ModelFreeRV) for anc in ancestors(vars)
    )


@node_rewriter([ModelFreeRV])
def reparam_loc_scale(fgraph, node):
    rv, value, *dims = node.inputs
    rv_node = rv.owner
    if rv_node is None:
        return None

    op = rv_node.op
    if type(op) not in LOC_SCALE_FAMILIES:
        return None

    size = op.size_param(rv_node)
    if isinstance(size.type, NoneTypeT):
        return None

    dist_params = list(op.dist_params(rv_node))
    loc_idx, scale_idx = LOC_SCALE_FAMILIES[type(op)]
    loc = dist_params[loc_idx]
    scale = dist_params[scale_idx]
    if not _is_broadcasted(rv, loc, scale):
        return None

    # Only reparametrize if loc or scale depends on another free RV
    if not _depends_on_free_rv([loc, scale]):
        return None

    rv_name = node.outputs[0].name or rv.name or "rv"
    rng = op.rng_param(rv_node)
    next_rng = op.update(rv_node)[rng]

    initial_point = getattr(fgraph, "_initial_point", None)
    if initial_point is None or value.name not in initial_point:
        return None

    # Create lam_logit with the dimensions shared by (loc, scale) & rv
    ndim = rv.type.ndim
    lam_bcast = pt.atleast_Nd(pt.broadcast_arrays(loc, scale)[0], n=ndim).type.broadcastable
    kept_axes = [i for i, bc in enumerate(lam_bcast) if not bc]
    dropped_axes = [i for i, bc in enumerate(lam_bcast) if bc]
    rv_shape = np.shape(initial_point[value.name])
    lam_shape = tuple(rv_shape[i] for i in kept_axes)
    lam_dims = tuple(dims[i] for i in kept_axes) if dims else ()

    lam_logit = pytensor.shared(
        np.zeros(lam_shape, dtype=rv.type.dtype),
        name=f"{rv_name}_lam_logit",
    )
    fgraph.add_input(lam_logit)
    lam_named = model_named(lam_logit, *lam_dims)
    lam = pt.sigmoid(lam_named)

    # Reinsert singleton axes at the collapsed positions so lam broadcasts
    # with loc/scale at the original ranks.
    if dropped_axes:
        lam = pt.expand_dims(lam, dropped_axes)

    new_params = list(dist_params)
    new_params[loc_idx] = lam * loc
    new_params[scale_idx] = scale**lam

    reparam_next_rng, reparam_rv = op.make_node(rng, size, *new_params).outputs
    reparam_name = f"{rv_name}_reparam"
    reparam_rv.name = reparam_name

    transform = node.op.transform
    value.name = get_transformed_name(reparam_name, transform) if transform else reparam_name
    reparam_free = model_free_rv(reparam_rv, value, transform, *dims)

    reparam_det = loc + scale ** (1 - lam) * (reparam_free - lam * loc)
    reparam_det.name = rv_name
    reparam_det = model_deterministic(reparam_det, *dims)
    return {node.outputs[0]: reparam_det, next_rng: reparam_next_rng}


reparam_rewrites_db = SequenceDB()
reparam_rewrites_db.name = "reparam_rewrites_db"
reparam_rewrites_db.register(
    "reparam_loc_scale",
    dfs_rewriter(reparam_loc_scale),
    "basic",
)


def automatic_reparametrization(
    model: Model,
    rewrite_db_query=RewriteDatabaseQuery(include=("basic",)),
    verbose: bool = False,
) -> tuple[Model, list[TensorSharedVariable]]:
    fgraph, _memo = fgraph_from_model(model)
    # Attach the initial point so rewrites can create valid initial shape for parameters
    fgraph._initial_point = model.initial_point()
    rewriter = reparam_rewrites_db.query(rewrite_db_query)
    _, _, rewrite_counters, *_ = rewriter.rewrite(fgraph)

    if verbose:
        applied = False
        for rewrite_counter in rewrite_counters:
            for rewrite, counts in rewrite_counter.items():
                applied = True
                print(f"Applied optimization: {rewrite} {counts}x", file=sys.stdout)  # noqa: T201
        if not applied:
            print("No optimizations applied", file=sys.stdout)  # noqa: T201

    reparam_model = model_from_fgraph(fgraph, mutate_fgraph=True)

    orig_data_vars = {d.name for d in model.data_vars}
    params = [d for d in reparam_model.data_vars if d.name not in orig_data_vars]

    return reparam_model, params
