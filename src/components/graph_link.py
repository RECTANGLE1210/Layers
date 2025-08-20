import torch
from torch import nn
from typing import Dict, Any, List

class GraphLink(nn.Module):
    """A module for defining a computation graph (DAG) with ParamVec operations.

    Args:
        nodes: List of node dicts with keys: id (str), op (str), inputs (List[str]), kwargs (dict).
            Each dict defines a node in the graph, where:
            - id: Unique identifier for the node.
            - op: Operation type (e.g., 'linear', 'add', 'mul', 'dot', 'concat', 'act', 'input', 'input_marker', 'output_marker', 'custom').
            - inputs: List of node IDs whose outputs feed into this node.
            - kwargs: Operation-specific parameters (e.g., weight, bias, dim_out, act_type).
        output_id: ID of the node whose output is the final result of the graph.
        trace: If True, prints debugging information (node ID, operation, and input shapes) during forward pass.

    Usage:
        Create a list of node dictionaries specifying the graph structure, then instantiate GraphLink with the nodes,
        output node ID, and optional trace flag. Example:
        nodes = [
            {"id": "input", "op": "input", "inputs": [], "kwargs": {}},
            {"id": "linear1", "op": "linear", "inputs": ["input"], "kwargs": {"dim_out": 4, "weight": ParamVec(), "bias": ParamVec()}},
            {"id": "softmax", "op": "act", "inputs": ["linear1"], "kwargs": {"act_type": "softmax"}}
        ]
        model = GraphLink(nodes, output_id="softmax", trace=True)
        output = model(torch.randn(8))  # Forward pass with input tensor
    """

    def __init__(self, nodes: List[Dict[str, Any]], output_id: str, trace: bool = False):
        """Initialize the GraphLink module by setting up nodes, output ID, and parameters.

        - Converts the list of nodes into a dictionary for O(1) access by node ID.
        - Computes topological order of nodes to ensure correct forward pass execution.
        - Registers trainable parameters (weights, biases, or other parameters) as PyTorch module parameters.
        - Parameters are registered with unique names based on node ID and parameter object ID to avoid conflicts.

        Usage:
            Instantiate with a list of node dictionaries and the output node ID. Set trace=True for debugging output during forward pass.
        """
        super().__init__()
        self.nodes = {node["id"]: node for node in nodes}
        self.output_id = output_id
        self.trace = trace
        self.topo_order = self._compute_topo_order()
        # Register parameters
        for node in nodes:
            if node["op"] in ["linear"]:
                self.add_module(f"weight_{node['id']}_{id(node['kwargs']['weight'])}", node["kwargs"]["weight"])
                if "bias" in node["kwargs"] and node["kwargs"]["bias"] is not None:
                    self.add_module(f"bias_{node['id']}_{id(node['kwargs']['bias'])}", node["kwargs"]["bias"])
            elif node["op"] in ["add", "mul", "dot"]:
                if "param" in node["kwargs"]:
                    self.add_module(f"param_{node['id']}_{id(node['kwargs']['param'])}", node["kwargs"]["param"])
            elif node["op"] == "concat":
                # concat now only uses previous node outputs, no params to register
                pass

    def _compute_topo_order(self) -> List[str]:
        """Compute the topological order of nodes to ensure dependency-respecting execution.

        - Uses depth-first search (DFS) to traverse the graph and produce a list of node IDs in topological order.
        - Detects cycles in the graph and raises a ValueError if found.
        - Returns a list of node IDs that can be used to process nodes in the correct order during the forward pass.

        Usage:
            Called internally during initialization. Not intended for direct use, but ensures forward pass processes nodes
            in an order where all inputs to a node are computed before the node itself.
        """
        graph = {nid: set(node["inputs"]) for nid, node in self.nodes.items()}
        result = []
        visited = set()
        temp_mark = set()

        def dfs(nid: str):
            if nid in temp_mark:
                raise ValueError("Graph has a cycle")
            if nid not in visited:
                temp_mark.add(nid)
                for dep in graph.get(nid, []):
                    dfs(dep)
                visited.add(nid)
                temp_mark.remove(nid)
                result.append(nid)

        for nid in self.nodes:
            if nid not in visited:
                dfs(nid)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass through the computation graph.

        Args:
            x: Input tensor to the graph.

        Returns:
            Output tensor from the node specified by output_id.

        - Processes nodes in topological order, computing each node's output based on its operation and inputs.
        - Supports operations: input, input_marker, output_marker, linear, add, mul, dot, concat, act, custom.
        - Automatically infers parameter shapes (e.g., weight, bias) if not specified, or validates specified shapes.
        - Raises ValueError for invalid shapes or unsupported operations.
        - If trace=True, prints node ID, operation, and input shapes for debugging.

        Usage:
            Pass an input tensor to compute the graph's output. Example:
            model = GraphLink(nodes, output_id="softmax")
            output = model(torch.randn(8))  # Computes output based on graph structure
        """
        values = {"input": x}
        for nid in self.topo_order:
            node = self.nodes[nid]
            op = node["op"]
            inputs = [values[inp] for inp in node["inputs"]]
            kwargs = node["kwargs"]
            if self.trace:
                print(f"Node {nid}: op={op}, input shapes={[t.shape for t in inputs]}")
            if op == "input":
                """Handle input node, which passes the input tensor directly.
                - Ensures only one node has op='input' and id='input'.
                - Stores the input tensor in values dictionary.
                """
                if nid != "input":
                    raise ValueError("Only one node can have op='input' with id='input'")
                values[nid] = x
            elif op == "input_marker":
                """Mark the input tensor without modification.
                - Useful for graphs where input needs to be referenced multiple times.
                - Stores the input tensor in values dictionary.
                """
                values[nid] = x
            elif op == "output_marker":
                """Mark the output of the graph.
                - Passes through the first input (or input tensor if no inputs).
                - Used to designate the final output node.
                """
                values[nid] = inputs[0] if inputs else x
            elif op == "linear":
                """Perform a linear transformation: y = xW^T + b.
                - Infers or validates weight shape as (dim_out, dim_in).
                - Infers or validates bias shape as (dim_out,) or (1,).
                - Raises ValueError if shapes are invalid.
                - Supports automatic shape materialization if shape is None.
                """
                dim_in = inputs[0].shape[-1] if inputs[0].dim() >= 1 else 1
                dim_out = kwargs["dim_out"]
                weight = kwargs["weight"]
                if weight.shape is None:
                    weight._ensure_materialized((dim_out, dim_in))
                else:
                    if weight.shape != (dim_out, dim_in):
                        raise ValueError(f"Invalid weight shape {weight.shape} for node {nid}, possible shapes: [({dim_out}, {dim_in})]")
                    weight._ensure_materialized((dim_out, dim_in))
                w_flat = weight.tensor()
                w_t = w_flat.view(dim_out, dim_in)
                result = torch.matmul(inputs[0], w_t.T)
                if "bias" in kwargs and kwargs["bias"] is not None:
                    bias = kwargs["bias"]
                    if bias.shape is None:
                        bias._ensure_materialized((dim_out,))
                    elif bias.shape == (dim_out,) :
                        bias._ensure_materialized((dim_out,))
                    elif bias.shape == (1,):
                        bias._ensure_materialized((1,))
                    else:
                        raise ValueError(f"Invalid bias shape {bias.shape} for node {nid}, possible shapes: [(1,), ({dim_out},)]")
                    b_t = bias.tensor()
                    result += b_t
                values[nid] = result
            elif op == "add":
                """Perform element-wise addition.
                - If 'param' is provided, adds a parameter tensor to the first input.
                - Otherwise, sums all input tensors (residual-style).
                - Infers or validates parameter shape as (dim,) or (1,).
                - Raises ValueError if shapes are invalid.
                """
                if "param" in kwargs:
                    param = kwargs["param"]
                    dim = inputs[0].shape[-1] if inputs[0].dim() >= 1 else 1
                    if param.shape is None:
                        param._ensure_materialized((dim,))
                    elif param.shape == (dim,):
                        param._ensure_materialized((dim,))
                    elif param.shape == (1,):
                        param._ensure_materialized((1,))
                    else:
                        raise ValueError(f"Invalid param shape {param.shape} for node {nid}, possible shapes: [(1,), ({dim},)]")
                    p_t = param.tensor()
                    values[nid] = inputs[0] + p_t
                else:
                    # Residual-style: sum all inputs
                    result = inputs[0]
                    for inp in inputs[1:]:
                        result = result + inp
                    values[nid] = result
            elif op == "mul":
                """Perform element-wise multiplication.
                - If 'param' is provided, multiplies the first input by a parameter tensor.
                - Otherwise, multiplies all input tensors together.
                - Infers or validates parameter shape as (dim,) or (1,).
                - Raises ValueError if shapes are invalid.
                """
                if "param" in kwargs:
                    param = kwargs["param"]
                    dim = inputs[0].shape[-1] if inputs[0].dim() >= 1 else 1
                    if param.shape is None:
                        param._ensure_materialized((dim,))
                    elif param.shape == (dim,):
                        param._ensure_materialized((dim,))
                    elif param.shape == (1,):
                        param._ensure_materialized((1,))
                    else:
                        raise ValueError(f"Invalid param shape {param.shape} for node {nid}, possible shapes: [(1,), ({dim},)]")
                    p_t = param.tensor()
                    values[nid] = inputs[0] * p_t
                else:
                    # Multiply all inputs together
                    result = inputs[0]
                    for inp in inputs[1:]:
                        result = result * inp
                    values[nid] = result
            elif op == "dot":
                """Perform dot product with a parameter tensor.
                - Computes sum(input * param) along the last dimension.
                - Infers or validates parameter shape as (dim,) or (1,).
                - Raises ValueError if shapes are invalid.
                - Output is a scalar or reduced tensor.
                """
                param = kwargs["param"]
                dim = inputs[0].shape[-1] if inputs[0].dim() >= 1 else 1
                if param.shape is None:
                    param._ensure_materialized((dim,))
                elif param.shape == (dim,):
                    param._ensure_materialized((dim,))
                elif param.shape == (1,):
                    param._ensure_materialized((1,))
                else:
                    raise ValueError(f"Invalid param shape {param.shape} for node {nid}, possible shapes: [(1,), ({dim},)]")
                p_t = param.tensor()
                values[nid] = torch.sum(inputs[0] * p_t, dim=-1)
            elif op == "concat":
                """Concatenate input tensors along the last dimension.
                - Combines outputs of previous nodes specified in inputs.
                - No parameters are required.
                - Output shape is the sum of input dimensions along the last axis.
                """
                values[nid] = torch.cat(inputs, dim=-1)
            elif op == "act":
                """Apply an activation function to the first input.
                - Supported activations: 'relu', 'tanh', 'softmax'.
                - Raises ValueError for unsupported act_type.
                - Output shape matches input shape (except softmax, which normalizes along last dimension).
                """
                act_type = kwargs["act_type"]
                if act_type == "relu":
                    values[nid] = torch.relu(inputs[0])
                elif act_type == "tanh":
                    values[nid] = torch.tanh(inputs[0])
                elif act_type == "softmax":
                    values[nid] = torch.softmax(inputs[0], dim=-1)
                else:
                    raise ValueError(f"Unsupported act_type: {act_type}")
            elif op == "custom":
                """Apply a custom function to the inputs.
                - Expects 'fn' in kwargs, a callable that processes the input list.
                - Output depends on the custom function's implementation.
                """
                fn = kwargs["fn"]
                values[nid] = fn(inputs)
            else:
                raise ValueError(f"Unsupported op: {op}")
        return values[self.output_id]