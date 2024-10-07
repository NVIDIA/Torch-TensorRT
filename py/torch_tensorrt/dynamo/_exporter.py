import base64
import copy
import operator
from typing import Any, Dict, Optional, Sequence, Tuple, cast

import torch
from torch._guards import detect_fake_mode
from torch._subclasses.fake_tensor import FakeTensor
from torch.export import ExportedProgram, ExportGraphSignature
from torch.export.exported_program import (
    CustomObjArgument,
    InputKind,
    InputSpec,
    ModuleCallEntry,
    ModuleCallSignature,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import ENGINE_IDX


def cross_save_for_windows(
    gm: torch.fx.GraphModule,
    file_path: str,
) -> None:
    gm = copy.deepcopy(gm)

    # Inline TensorRT submodules for windows
    inline_trt_modules_for_windows(gm)

    # Inline pytorch submodules
    inline_torch_modules(gm)

    # Clean the graph
    gm.delete_all_unused_submodules()
    gm.graph.eliminate_dead_code()
    gm.graph.lint()

    exp_program = create_trt_exp_program(gm)
    torch.export.save(exp_program, file_path)


def export(
    gm: torch.fx.GraphModule,
    inputs: Sequence[torch.Tensor],
    kwarg_inputs: Optional[dict[str, Any]] = None,
) -> ExportedProgram:
    """Export the result of TensorRT compilation into the desired output format.

    Arguments:
        gm (torch.fx.GraphModule): Compiled Torch-TensorRT module, generated by ``torch_tensorrt.dynamo.compile``
        inputs (torch.Tensor): Torch input tensors
    """
    if kwarg_inputs is None:
        kwarg_inputs = {}
    patched_module = transform(gm, inputs, kwarg_inputs)
    exp_program = create_trt_exp_program(patched_module)
    return exp_program


def transform(
    gm: torch.fx.GraphModule,
    inputs: Sequence[torch.Tensor],
    kwarg_inputs: Optional[dict[str, Any]] = None,
) -> torch.fx.GraphModule:
    """
    Transforms the graphmodule by inlining Pytorch and TensorRT submodules.
    Inlining collapses submodules into nodes which is necessary for torch.export
    serialization.

    Arguments:
        gm (torch.fx.GraphModule): Compiled Torch-TensorRT module, generated by ``torch_tensorrt.dynamo.compile``
        inputs (torch.Tensor): Torch input tensors

    Returns an inlined torch.fx.GraphModule
    """
    # Make a copy the graph since this function transforms the input graph and changes it's attributes.
    # This transformed graph is meant to be consumed by `create_trt_exp_program`
    if kwarg_inputs is None:
        kwarg_inputs = {}
    gm = copy.deepcopy(gm)

    # Inline TensorRT submodules
    inline_trt_modules(gm)

    # Inline pytorch submodules
    inline_torch_modules(gm)

    # Clean the graph
    gm.delete_all_unused_submodules()
    gm.graph.eliminate_dead_code()
    gm.graph.lint()

    return gm


def lift(
    gm: torch.fx.GraphModule, graph_signature: Any
) -> Tuple[torch.fx.GraphModule, ExportGraphSignature, Dict[str, Any], Dict[str, Any]]:
    """
    Given an unlifted fx.GraphModule, lift all parameters, buffers into placeholders.
    Arguments:
        gm (torch.fx.GraphModule): Unlifted GraphModule which contains parameters and buffers as get_attr nodes.
        graph_signature (torch.export.ExportGraphSignature): Instance of ExportGraphSignature class created for the output ExportedProgram.
        After lifting, this graph_signature will be modified with the parameters and buffers added appropriately.
    Returns:
        A lifted fx.GraphModule, modified graph_signature and a new state_dict
    """
    # Get the state_dict of graph_module. This is different from exported_program.state_dict
    # exp_program.state_dict contains parameters and buffers whereas a graph_module's state_dict
    # has all parameters registered as torch.tensors.
    state_dict = gm.state_dict()
    constants = {}

    fake_mode = detect_fake_mode(
        tuple(node.meta["val"] for node in gm.graph.nodes if node.op == "placeholder")
    )
    assert fake_mode is not None

    # This map stores the names of outputs (old to new)
    # This is necessary to track because the output names can be changed when
    # we convert graph constants to placeholder inputs below.
    output_names = {}
    for output_spec in graph_signature.output_specs:
        output_names[output_spec.arg.name] = output_spec.arg.name

    # Locate the user input to insert new placeholders before them
    first_user_input = None
    for node in gm.graph.nodes:
        if node.op == "placeholder" and node.name in graph_signature.user_inputs:
            first_user_input = node
            break

    # At first the user_inputs are only present in the graph_signature.input_specs and hence non_user_input_idx=0
    # The input_specs should be of the form [params, buffers, constant_tensors, custom_obj, user_inputs]
    non_user_input_idx = 0
    for node in gm.graph.nodes:
        if node.op == "get_attr":

            lift_val = None
            input_kind = None

            if node.target not in state_dict:
                constants[node.target] = getattr(gm, node.target)
                input_kind = InputKind.CUSTOM_OBJ
                lift_val = constants[node.target]
            else:
                lift_val = state_dict[node.target]

                input_kind = InputKind.CONSTANT_TENSOR

                # state_dict has these parameters/buffers as torch.Tensors. We override them as torch.nn.Parameter/torch.Tensors respectively.
                for name, _ in gm.named_parameters():
                    if node.target == name:
                        input_kind = InputKind.PARAMETER
                        state_dict[name] = torch.nn.Parameter(state_dict[name])
                        break
                for name, _ in gm.named_buffers():
                    if node.target == name:
                        input_kind = InputKind.BUFFER
                        break

            assert lift_val is not None and input_kind is not None

            # Replace get_attr nodes with placeholder nodes and copy metadata.
            with gm.graph.inserting_before(first_user_input):
                # Ensure name doesn't contain period as it is used for submodules
                const_placeholder_name = node.target.replace(".", "_")
                const_placeholder_node = gm.graph.placeholder(const_placeholder_name)
                # Copy the node meta into this new placeholder node
                const_placeholder_node.meta = node.meta

                if isinstance(lift_val, torch.Tensor):
                    const_placeholder_node.meta["val"] = cast(
                        FakeTensor,
                        torch.empty_strided(
                            tuple(lift_val.shape),
                            tuple([1] * len(lift_val.shape)),
                        ),
                    )

                node.replace_all_uses_with(const_placeholder_node)
                gm.graph.erase_node(node)

                # Verify if the const_placeholder being added is one of the output nodes
                # This happens if there is just a single static arange op in the graph
                # https://github.com/pytorch/TensorRT/issues/3189
                if const_placeholder_name in output_names:
                    output_names[const_placeholder_name] = const_placeholder_node.name

                # Add these parameters/buffers/constants to the existing graph signature
                # before user inputs. These specs are looked up in the state_dict during ExportedProgram creation.
                input_spec_arg = TensorArgument(name=const_placeholder_node.name)
                if input_kind == InputKind.CUSTOM_OBJ:
                    input_spec_arg = CustomObjArgument(
                        name=const_placeholder_node.name, class_fqn=""
                    )
                graph_signature.input_specs.insert(
                    non_user_input_idx,
                    InputSpec(
                        kind=input_kind,
                        arg=input_spec_arg,
                        target=node.target,
                    ),
                )
                non_user_input_idx += 1

    # Update output_specs with modified names. This only gets updated if the graph getattr nodes (weights)
    # are also the outputs of the graph
    for output_spec in graph_signature.output_specs:
        output_spec.arg.name = output_names[output_spec.arg.name]

    gm.graph.eliminate_dead_code()
    gm.graph.lint()

    return gm, graph_signature, state_dict, constants


def get_duplicate_nodes(
    gm: torch.fx.GraphModule, submodule: torch.fx.GraphModule
) -> Tuple[Sequence[Any], Sequence[Any]]:
    """
    We check if there are duplicate nodes when we copy submodule graph into gm.
    Handle the case where the subgraph input placeholders are same as
    gm placeholders. This happens when the first submodule in the graph is
    a pytorch submodule
    """
    submodule_placeholder_inputs = [
        node for node in submodule.graph.nodes if node.op == "placeholder"
    ]
    submodule_input_node_names = [node.name for node in submodule_placeholder_inputs]
    gm_node_names = [node.name for node in gm.graph.nodes]
    submodule_duplicate_inputs = [
        node for node in submodule_placeholder_inputs if node.name in gm_node_names
    ]
    gm_duplicate_inputs = [
        node for node in gm.graph.nodes if node.name in submodule_input_node_names
    ]
    return submodule_duplicate_inputs, gm_duplicate_inputs


def inline_torch_modules(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Inline a submodule within the parent graph (gm). All `call_module` nodes
    should be replaced by their nodes in the submodule.
    """
    # Clean the graph
    gm.graph.eliminate_dead_code()
    gm.graph.lint()

    for gm_node in gm.graph.nodes:
        if gm_node.op == "call_module" and "_run_on_gpu" in gm_node.name:
            submodule = getattr(gm, gm_node.name)
            with gm.graph.inserting_before(gm_node):
                # Get inputs of submodule node which are most likely outputs of a previous TRT node
                # or a placeholder of the main graph
                submodule_inputs = gm_node.args

                submodule_duplicate_inputs, gm_duplicate_inputs = get_duplicate_nodes(
                    gm, submodule
                )
                assert len(submodule_duplicate_inputs) == len(gm_duplicate_inputs)
                # Avoid creating new copies of duplicate inputs by creating a mapping
                val_map = {}
                for i in range(len(submodule_duplicate_inputs)):
                    val_map[submodule_duplicate_inputs[i]] = gm_duplicate_inputs[i]

                # Copy all nodes in the submodule into gm and
                # store the output node of this submodule which is now present in gm
                submodule_output = gm.graph.graph_copy(submodule.graph, val_map)

                # Get their references (since we copied) in the parent graph (gm)
                if len(submodule_duplicate_inputs) == 0:
                    submodule_placeholder_input_names = [
                        node.name
                        for node in submodule.graph.nodes
                        if node.op == "placeholder"
                    ]
                    gm_added_placeholder_inputs = [
                        node
                        for node in gm.graph.nodes
                        if node.name in submodule_placeholder_input_names
                    ]

                    assert len(submodule_inputs) == len(gm_added_placeholder_inputs)

                    # Replace the added placeholder inputs with original inputs to this submodule node
                    for idx in range(len(gm_added_placeholder_inputs)):
                        gm_added_placeholder_inputs[idx].replace_all_uses_with(
                            submodule_inputs[idx]
                        )

                    # Erase the placeholder input nodes in the gm
                    for idx in range(len(gm_added_placeholder_inputs)):
                        gm.graph.erase_node(gm_added_placeholder_inputs[idx])

                # Replace the pytorch submodule node (call_module) with the inlined subgraph output
                gm_node.replace_all_uses_with(submodule_output)

                # copy the attributes of the submodule into gm (graph_copy doesn't do this)
                copy_submodule_attributes(gm, submodule, gm_node.name)

            # Erase the pytorch submodule (call_module) node
            gm.graph.erase_node(gm_node)

    return gm


def copy_submodule_attributes(
    gm: torch.fx.GraphModule, submodule: torch.fx.GraphModule, submodule_name: str
) -> None:
    """
    The submodule parameters are available in the parent gm's state_dict, but they have
    the submodule name as a prefix in their keys. For eg: gm.state_dict() would have
    _run_on_gpu_0.conv.weight etc. Since we graph copied the submodule into gm, we should
    also copy it's parameters and buffers into gm without the submodule namespace as prefix.
    _assign_attr does exactly that. It creates a module for eg: conv, adds an attribute weight
    to it and adds this conv module as an attribute to parent gm.
    """
    from torch.export.unflatten import _assign_attr, _AttrKind

    for key, value in submodule.named_parameters():
        _assign_attr(value, gm, key, _AttrKind.PARAMETER)

    for key, value in submodule.named_buffers():
        _assign_attr(value, gm, key, _AttrKind.BUFFER)


def create_trt_exp_program(
    gm: torch.fx.GraphModule,
) -> ExportedProgram:
    """Creates a new Exported Program. This function takes an torch.fx.GraphModule which has TRT engines
    and constructs an Exported Program object with the new IO node names and state_dict
    """

    input_nodes = [node for node in gm.graph.nodes if node.op == "placeholder"]
    output_nodes = [node for node in gm.graph.nodes if node.op == "output"]
    assert output_nodes
    output_nodes = output_nodes[0].args[0]

    input_specs = [
        InputSpec(InputKind.USER_INPUT, TensorArgument(name=node.name), node.target)
        for node in input_nodes
    ]
    output_specs = [
        OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(name=node.name), node.target)
        for node in output_nodes
    ]

    trt_graph_signature = ExportGraphSignature(
        input_specs=input_specs, output_specs=output_specs
    )

    module_call_graph = [
        ModuleCallEntry(
            "",
            ModuleCallSignature(
                inputs=[],
                outputs=[],
                in_spec=gm.graph._codegen.pytree_info.in_spec,
                out_spec=gm.graph._codegen.pytree_info.out_spec,
            ),
        )
    ]

    # Lift parameters/buffers/constants in the graph
    # torch.export serialization expects them to be lifted
    gm, trt_graph_signature, state_dict, constants = lift(gm, trt_graph_signature)

    trt_exp_program = ExportedProgram(
        root=gm,
        graph=gm.graph,
        graph_signature=trt_graph_signature,
        state_dict=state_dict,
        range_constraints={},
        module_call_graph=module_call_graph,
        constants=constants,
    )

    return trt_exp_program


def inline_trt_modules(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Replace TRT submodules with trt engine nodes.
    """
    for name, _ in gm.named_children():
        if "_run_on_acc" not in name:
            continue
        # Get the TRT submodule
        trt_module = getattr(gm, name)

        # Ensure the trt module node in the main graph (gm) has inputs
        trt_module_node = [node for node in gm.graph.nodes if node.name == name]
        assert trt_module_node
        trt_module_node = trt_module_node[0]
        assert trt_module_node.args

        num_outputs = len(trt_module.output_shapes)
        # Insert a call_function node to perform inference on TRT engine
        with gm.graph.inserting_before(trt_module_node):
            engine_name = f"{name}_engine"
            setattr(gm, engine_name, trt_module.engine)
            engine_node = gm.graph.get_attr(engine_name)

            trt_node = gm.graph.call_function(
                torch.ops.tensorrt.execute_engine.default,
                (trt_module_node.args, engine_node),
            )
            trt_node.meta["val"] = []
            assert num_outputs > 0
            # Generate meta data for TRT node (a FakeTensor with corresponding output shape)
            for idx in range(num_outputs):
                trt_node.meta["val"].append(
                    cast(
                        FakeTensor,
                        torch.empty_strided(
                            tuple(trt_module.output_shapes[idx]),
                            tuple([1] * len(trt_module.output_shapes[idx])),
                        ),
                    )
                )

            # meta["val"] should be a lighter version of a tensor. For eg: it should be a FakeTensor (with output shape and dtype properties)
            # Lighter version of a custom_obj is not defined clearly. meta["val"] does not have any type expectations but
            # for custom object nodes, it should be CustomObjArgument
            engine_node.meta["val"] = CustomObjArgument(
                name=engine_node.name, class_fqn=""
            )

        if num_outputs == 1:
            # Insert getitem nodes as outputs (for export serialization to work)
            with gm.graph.inserting_after(trt_node):
                getitem_output = gm.graph.call_function(operator.getitem, (trt_node, 0))
                getitem_output.meta["val"] = trt_node.meta["val"]
            trt_module_node.replace_all_uses_with(getitem_output)
        else:
            # Multiple outputs case:
            # Replace uses of submodule with the trt_node.
            # getitem nodes are already added inherently by the partitioner
            trt_module_node.replace_all_uses_with(trt_node)
            getitem_nodes = trt_node.users
            for idx, getitem_node in enumerate(getitem_nodes):
                getitem_node.meta["val"] = trt_node.meta["val"][idx]

        # Erase the TRT submodule (call_module) node.
        gm.graph.erase_node(trt_module_node)

    return gm


def inline_trt_modules_for_windows(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Replace TRT submodules with trt engine nodes.
    """
    for name, _ in gm.named_children():
        if "_run_on_acc" not in name:
            continue
        # Get the TRT submodule
        trt_module = getattr(gm, name)

        # Ensure the trt module node in the main graph (gm) has inputs
        trt_module_node = [node for node in gm.graph.nodes if node.name == name]
        assert trt_module_node
        trt_module_node = trt_module_node[0]
        assert trt_module_node.args

        num_outputs = len(trt_module.output_shapes)
        # Insert a call_function node to perform inference on TRT engine
        with gm.graph.inserting_before(trt_module_node):
            engine_info = trt_module._pack_engine_info()
            engine_bytes = engine_info[ENGINE_IDX]
            engine_info[ENGINE_IDX] = base64.b64encode(engine_bytes).decode("utf-8")

            engine_node = gm.graph.call_function(
                torch.ops.tensorrt.setup_engine.default,
                tuple(engine_info),
            )

            trt_node = gm.graph.call_function(
                torch.ops.tensorrt.execute_engine.default,
                (trt_module_node.args, engine_node),
            )

            trt_node.meta["val"] = []
            assert num_outputs > 0
            # Generate meta data for TRT node (a FakeTensor with corresponding output shape)
            for idx in range(num_outputs):
                trt_node.meta["val"].append(
                    cast(
                        FakeTensor,
                        torch.empty_strided(
                            tuple(trt_module.output_shapes[idx]),
                            tuple([1] * len(trt_module.output_shapes[idx])),
                        ),
                    )
                )
            # Generate meta data for engine_node (a FakeTensor with corresponding output shape)
            engine_node.meta["val"] = cast(
                FakeTensor,
                torch.empty_strided(
                    tuple(trt_module.output_shapes[0]),
                    tuple([1] * len(trt_module.output_shapes[0])),
                ),
            )

        if num_outputs == 1:
            # Insert getitem nodes as outputs (for export serialization to work)
            with gm.graph.inserting_after(trt_node):
                getitem_output = gm.graph.call_function(operator.getitem, (trt_node, 0))
                getitem_output.meta["val"] = trt_node.meta["val"]
            trt_module_node.replace_all_uses_with(getitem_output)
        else:
            # Multiple outputs case:
            # Replace uses of submodule with the trt_node.
            # getitem nodes are already added inherently by the partitioner
            trt_module_node.replace_all_uses_with(trt_node)
            getitem_nodes = trt_node.users
            for idx, getitem_node in enumerate(getitem_nodes):
                getitem_node.meta["val"] = trt_node.meta["val"][idx]

        # Erase the TRT submodule (call_module) node.
        gm.graph.erase_node(trt_module_node)

    return gm
