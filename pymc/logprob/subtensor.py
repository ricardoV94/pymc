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
from typing import Optional

from pytensor import tensor as pt
from pytensor.graph import node_rewriter
from pytensor.tensor import broadcast_arrays
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.random.rewriting import local_subtensor_rv_lift
from pytensor.tensor.rewriting.subtensor import local_subtensor_lift
from pytensor.tensor.subtensor import AdvancedSubtensor, AdvancedSubtensor1

from pymc.logprob.rewriting import (
    PreserveRVMappings,
    inc_subtensor_ops,
    subtensor_ops,
    measurable_ir_rewrites_db,
)
from pymc.logprob.utils import indices_from_subtensor


@node_rewriter(inc_subtensor_ops)
def incsubtensor_rv_replace(fgraph, node):
    r"""Replace `*IncSubtensor*` `Op`\s and their value variables for log-probability calculations.

    This is used to derive the log-probability graph for ``Y[idx] = data``, where
    ``Y`` is a `RandomVariable`, ``idx`` indices, and ``data`` some arbitrary data.

    To compute the log-probability of a statement like ``Y[idx] = data``, we must
    first realize that our objective is equivalent to computing ``logprob(Y, z)``,
    where ``z = pt.set_subtensor(y[idx], data)`` and ``y`` is the value variable
    for ``Y``.

    In other words, the log-probability for an `*IncSubtensor*` is the log-probability
    of the underlying `RandomVariable` evaluated at ``data`` for the indices
    given by ``idx`` and at the value variable for ``~idx``.

    This provides a means of specifying "missing data", for instance.
    """
    rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, "preserve_rv_mappings", None)

    if rv_map_feature is None:
        return None  # pragma: no cover

    rv_var = node.outputs[0]
    if rv_var not in rv_map_feature.rv_values:
        return None  # pragma: no cover

    base_rv_var = node.inputs[0]

    if not rv_map_feature.request_measurable([base_rv_var]):
        return None

    data = node.inputs[1]
    idx = indices_from_subtensor(getattr(node.op, "idx_list", None), node.inputs[2:])

    # Create a new value variable with the indices `idx` set to `data`
    value_var = rv_map_feature.rv_values[rv_var]
    new_value_var = pt.set_subtensor(value_var[idx], data)
    rv_map_feature.update_rv_maps(rv_var, new_value_var, base_rv_var)

    # Return the `RandomVariable` being indexed
    return [base_rv_var]


@node_rewriter(subtensor_ops)
def elemwise_subtensor_lift(fgraph, node):
    """Lift subtensor Ops through Elemwise operations"""
    rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, "preserve_rv_mappings", None)

    if rv_map_feature is None:
        return None  # pragma: no cover

    op = node.op
    base_var, *raw_indices = node.inputs

    # Don't lift through valued RVs
    if base_var in rv_map_feature.rv_values:
        return None

    base_node = base_var.owner
    if not isinstance(base_node.op, Elemwise):
        return None

    indices = indices_from_subtensor(getattr(node.op, "idx_list", None), raw_indices)
    # Only allow slice or boolean AdvancedSubtensors (integers can broadcast)
    # TODO: We could allow scalar integers
    if isinstance(op, (AdvancedSubtensor, AdvancedSubtensor1)) and any(index.dtype == int for index in indices):
        return None

    non_scalar_input_idxs = [idx for idx, inp in enumerate(base_node.inputs) if any(s != 1 for s in inp.type.shape)]
    if not non_scalar_input_idxs:
        # If there are no non_scalar_inputs, pass the index to the first input
        non_scalar_input_idxs = [0]
        bcast_inputs = [base_node.inputs[0]]
    elif len(non_scalar_input_idxs) == 1:
        bcast_inputs = [base_node.inputs[non_scalar_input_idxs[0]]]
    else:
        bcast_inputs = broadcast_arrays(
            *(
                inp for idx, inp in enumerate(base_node.inputs)
                if idx in non_scalar_input_idxs
            )
        )

    new_inputs = []
    bcast_input_idx = 0
    for idx, inp in enumerate(base_node.inputs):
        # We pass the index to the bcast inputs only
        if idx in non_scalar_input_idxs:
            bcast_inp = bcast_inputs[bcast_input_idx]
            # If the input was not actually broadcasted, use the original one
            if inp.type.broadcastable == bcast_inp.type.broadcastable:
                bcast_inp = inp
            new_inputs.append(bcast_inp[indices])
            bcast_input_idx += 1
        # Others are squeezed
        else:
            new_inputs.append(pt.squeeze(inp))

    return base_node.op.make_node(*new_inputs).outputs


# These rewrites push random/measurable variables "down", making them closer to
# (or eventually) the graph outputs.  Often this is done by lifting other `Op`s
# "up" through the random/measurable variables and into their inputs.
measurable_ir_rewrites_db.register("elemwise_subtensor_lift", elemwise_subtensor_lift, "basic")
measurable_ir_rewrites_db.register("rv_subtensor_lift", local_subtensor_rv_lift, "basic")
measurable_ir_rewrites_db.register("incsubtensor_lift", incsubtensor_rv_replace, "basic")
