import logging
from typing import Sequence

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings

# dead-code elimination, linting, and recompilation for graph, in-place
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def custom_fused_all_gather_op(args0, args1, args2):
    return torch.ops._c10d_functional.wait_tensor.default(
        torch.ops._c10d_functional.all_gather_into_tensor.default(args0, args1, args2)
    )


def custom_fused_reduce_scatter_op(args0, args1, args2, args3):
    return torch.ops._c10d_functional.wait_tensor.default(
        torch.ops._c10d_functional.reduce_scatter_tensor.default(
            args0, args1, args2, args3
        )
    )


def fuse_distributed_ops(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    modified_graph = False
    for node in gm.graph.nodes:
        if (
            node.target
            in (
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
            )
            and len(node.users) == 1
            and list(node.users)[0].target
            == torch.ops._c10d_functional.wait_tensor.default
        ):
            wait_tensor_node = list(node.users)[0]
            fused_op = None
            if node.target == torch.ops._c10d_functional.all_gather_into_tensor.default:
                fused_op = custom_fused_all_gather_op
                fused_op_args = (node.args[0], node.args[1], node.args[2])
            else:
                fused_op = custom_fused_reduce_scatter_op
                fused_op_args = (node.args[0], node.args[1], node.args[2], node.args[3])
            with gm.graph.inserting_after(wait_tensor_node):
                fused_node = gm.graph.create_node(
                    op="call_function",
                    target=fused_op,  # Define your custom fused function
                    args=fused_op_args,
                )

            wait_tensor_node.replace_all_uses_with(fused_node)
            fused_node.meta.update(node.meta)
            modified_graph = True
            gm.graph.erase_node(wait_tensor_node)
            gm.graph.erase_node(node)

    # If graph was modified, clean it up
    if modified_graph:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(
            f"Graph after fusing wait_tensor and distributed op tensor:\n{gm.graph}"
        )

    return gm
