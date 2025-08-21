import torch
from torch import nn
from typing import Dict, Any, List, Optional

class Node(nn.Module):
    """Base class for defining a single node in a computation graph.

    Args:
        id: Unique identifier for the node.
        op: Operation type (e.g., 'linear', 'add', 'mul', 'dot', 'concat', 'act', 'input', 'input_marker', 'output_marker', 'custom').
        inputs: List of node IDs whose outputs feed into this node.
        kwargs: Operation-specific parameters (e.g., weight, bias, dim_out, act_type).

    Usage:
        Subclass Node to implement specific operations or instantiate directly:
        node = Node(id="linear1", op="linear", inputs=["input"], kwargs={"dim_out": 4, "weight": ParamVec(), "bias": ParamVec()})
    """
    def __init__(self, id: str, op: str, inputs: List[str], kwargs: Dict[str, Any]):
        super().__init__()
        self.id = id
        self.op = op
        self.inputs = inputs
        self.kwargs = kwargs
        self.input_shape = None
        self.output_shape = None
        self._register_parameters()

    def _register_parameters(self):
        """Register node parameters as PyTorch module parameters."""
        if self.op == "linear":
            self.add_module(f"weight_{self.id}_{id(self.kwargs['weight'])}", self.kwargs["weight"])
            if "bias" in self.kwargs and self.kwargs["bias"] is not None:
                self.add_module(f"bias_{self.id}_{id(self.kwargs['bias'])}", self.kwargs["bias"])
        elif self.op in ["add", "mul", "dot"]:
            if "param" in self.kwargs:
                self.add_module(f"param_{self.id}_{id(self.kwargs['param'])}", self.kwargs["param"])
        elif self.op == "concat":
            # concat uses no parameters
            pass

    def forward(self, input_tensors: List[torch.Tensor], x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the node's output based on its operation and inputs.

        Args:
            input_tensors: List of input tensors from parent nodes.
            x: Input tensor for 'input' or 'input_marker' operations (optional).

        Returns:
            Output tensor of the node.
        """
        try:
            if self.op == "input":
                if self.id != "input":
                    raise ValueError("Only one node can have op='input' with id='input'")
                result = x
                self.input_shape = None
                self.output_shape = tuple(result.shape) if result is not None else None
                return result
            elif self.op == "input_marker":
                result = x
                self.input_shape = None
                self.output_shape = tuple(result.shape) if result is not None else None
                return result
            elif self.op == "output_marker":
                result = input_tensors[0] if input_tensors else x
                self.input_shape = [t.shape for t in input_tensors] if input_tensors else None
                self.output_shape = tuple(result.shape) if result is not None else None
                return result
            elif self.op == "linear":
                dim_in = input_tensors[0].shape[-1] if input_tensors[0].dim() >= 1 else 1
                dim_out = self.kwargs["dim_out"]
                weight = self.kwargs["weight"]
                if weight.shape is None:
                    weight._ensure_materialized((dim_out, dim_in))
                else:
                    if weight.shape != (dim_out, dim_in):
                        raise ValueError(f"Invalid weight shape {weight.shape} for node {self.id}, possible shapes: [({dim_out}, {dim_in})]")
                    weight._ensure_materialized((dim_out, dim_in))
                w_flat = weight.tensor()
                w_t = w_flat.view(dim_out, dim_in)
                result = torch.matmul(input_tensors[0], w_t.T)
                if "bias" in self.kwargs and self.kwargs["bias"] is not None:
                    bias = self.kwargs["bias"]
                    if bias.shape is None:
                        bias._ensure_materialized((dim_out,))
                    elif bias.shape == (dim_out,) or bias.shape == (1,):
                        bias._ensure_materialized(bias.shape)
                    else:
                        raise ValueError(f"Invalid bias shape {bias.shape} for node {self.id}, possible shapes: [(1,), ({dim_out},)]")
                    b_t = bias.tensor()
                    result += b_t
                self.input_shape = [t.shape for t in input_tensors]
                self.output_shape = tuple(result.shape)
                return result
            elif self.op == "add":
                if "param" in self.kwargs:
                    param = self.kwargs["param"]
                    dim = input_tensors[0].shape[-1] if input_tensors[0].dim() >= 1 else 1
                    if param.shape is None:
                        param._ensure_materialized((dim,))
                    elif param.shape == (dim,) or param.shape == (1,):
                        param._ensure_materialized(param.shape)
                    else:
                        raise ValueError(f"Invalid param shape {param.shape} for node {self.id}, possible shapes: [(1,), ({dim},)]")
                    p_t = param.tensor()
                    result = input_tensors[0] + p_t
                else:
                    result = input_tensors[0]
                    for inp in input_tensors[1:]:
                        result = result + inp
                self.input_shape = [t.shape for t in input_tensors]
                self.output_shape = tuple(result.shape)
                return result
            elif self.op == "mul":
                if "param" in self.kwargs:
                    param = self.kwargs["param"]
                    dim = input_tensors[0].shape[-1] if input_tensors[0].dim() >= 1 else 1
                    if param.shape is None:
                        param._ensure_materialized((dim,))
                    elif param.shape == (dim,) or param.shape == (1,):
                        param._ensure_materialized(param.shape)
                    else:
                        raise ValueError(f"Invalid param shape {param.shape} for node {self.id}, possible shapes: [(1,), ({dim},)]")
                    p_t = param.tensor()
                    result = input_tensors[0] * p_t
                else:
                    result = input_tensors[0]
                    for inp in input_tensors[1:]:
                        result = result * inp
                self.input_shape = [t.shape for t in input_tensors]
                self.output_shape = tuple(result.shape)
                return result
            elif self.op == "dot":
                param = self.kwargs["param"]
                dim = input_tensors[0].shape[-1] if input_tensors[0].dim() >= 1 else 1
                if param.shape is None:
                    param._ensure_materialized((dim,))
                elif param.shape == (dim,) or param.shape == (1,):
                    param._ensure_materialized(param.shape)
                else:
                    raise ValueError(f"Invalid param shape {param.shape} for node {self.id}, possible shapes: [(1,), ({dim},)]")
                p_t = param.tensor()
                result = torch.sum(input_tensors[0] * p_t, dim=-1)
                self.input_shape = [t.shape for t in input_tensors]
                self.output_shape = tuple(result.shape)
                return result
            elif self.op == "concat":
                result = torch.cat(input_tensors, dim=-1)
                self.input_shape = [t.shape for t in input_tensors]
                self.output_shape = tuple(result.shape)
                return result
            elif self.op == "act":
                act_type = self.kwargs["act_type"]
                if act_type == "relu":
                    result = torch.relu(input_tensors[0])
                elif act_type == "tanh":
                    result = torch.tanh(input_tensors[0])
                elif act_type == "softmax":
                    result = torch.softmax(input_tensors[0], dim=-1)
                else:
                    raise ValueError(f"Unsupported act_type: {act_type}")
                self.input_shape = [t.shape for t in input_tensors]
                self.output_shape = tuple(result.shape)
                return result
            elif self.op == "custom_func":
                fn = self.kwargs["fn"]
                if isinstance(fn, nn.Module) and any(p.requires_grad for p in fn.parameters()):
                    raise ValueError("custom_func must not be an nn.Module with parameters. Use custom_block instead.")
                result = fn(input_tensors if len(input_tensors) > 1 else input_tensors[0])
                self.input_shape = [t.shape for t in input_tensors]
                self.output_shape = tuple(result.shape)
                return result
            elif self.op == "custom_block":
                block = self.kwargs["fn"]
                if not isinstance(block, nn.Module):
                    raise ValueError("custom_block requires an nn.Module instance")
                if not hasattr(self, "_custom_registered"):
                    self.add_module(f"custom_{self.id}", block)
                    self._custom_registered = True
                result = block(input_tensors[0] if len(input_tensors) == 1 else input_tensors)
                self.input_shape = [t.shape for t in input_tensors]
                self.output_shape = tuple(result.shape)
                return result
            elif self.op == "avg":
                # tính trung bình các input tensors
                result = input_tensors[0]
                for inp in input_tensors[1:]:
                    result = result + inp
                result = result / len(input_tensors)
                self.input_shape = [t.shape for t in input_tensors]
                self.output_shape = tuple(result.shape)
                return result
            elif self.op == "sign":
                result = torch.sign(input_tensors[0])
                self.input_shape = [t.shape for t in input_tensors]
                self.output_shape = tuple(result.shape)
                return result
            elif self.op == "threshold":
                threshold = self.kwargs.get("threshold", 0.5)
                result = (input_tensors[0] > threshold).to(input_tensors[0].dtype)
                self.input_shape = [t.shape for t in input_tensors]
                self.output_shape = tuple(result.shape)
                return result
            elif self.op == "argmax":
                result = torch.argmax(input_tensors[0], dim=-1)
                self.input_shape = [t.shape for t in input_tensors]
                self.output_shape = tuple(result.shape)
                return result
            elif self.op == "argmin":
                result = torch.argmin(input_tensors[0], dim=-1)
                self.input_shape = [t.shape for t in input_tensors]
                self.output_shape = tuple(result.shape)
                return result
            else:
                raise ValueError(f"Unsupported op: {self.op}")
        except Exception as e:
            raise RuntimeError(f"[GraphLink] Error in node '{self.id}' (op={self.op}): {str(e)}") from e

class GraphLink(nn.Module):
    """A module for defining a computation graph (DAG) with Node operations.

    Args:
        nodes: List of Node objects defining the graph structure.
        output_id: ID of the node whose output is the final result of the graph.
        trace: If True, prints debugging information (node ID, operation, and input shapes) during forward pass.

    Usage:
        Create Node objects and pass them to GraphLink:
        nodes = [
            Node(id="input", op="input", inputs=[], kwargs={}),
            Node(id="linear1", op="linear", inputs=["input"], kwargs={"dim_out": 4, "weight": ParamVec(), "bias": ParamVec()}),
            Node(id="softmax", op="act", inputs=["linear1"], kwargs={"act_type": "softmax"})
        ]
        model = GraphLink(nodes, output_id="softmax", trace=True)
        output = model(torch.randn(8))  # Forward pass with input tensor
    """
    def __init__(self, nodes: List[Node], output_id: str, trace: bool = False):
        super().__init__()
        node_ids = [node.id for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError(f"Duplicate node IDs found: {node_ids}")
        self.nodes = {node.id: node for node in nodes}
        self.output_id = output_id
        self.trace = trace
        self.topo_order = self._compute_topo_order()
        for node in nodes:
            self.add_module(f"node_{node.id}", node)

    def _compute_topo_order(self) -> List[str]:
        """Compute the topological order of nodes to ensure dependency-respecting execution."""
        graph = {node.id: set(node.inputs) for node in self.nodes.values()}
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
        """
        values = {"input": x}
        for nid in self.topo_order:
            node = self.nodes[nid]
            inputs = [values[inp] for inp in node.inputs]
            if self.trace:
                print(f"Node {nid}: op={node.op}, input shapes={[t.shape for t in inputs]}")
            try:
                values[nid] = node.forward(inputs, x)
            except Exception as e:
                raise RuntimeError(f"[GraphLink] Shape mismatch at node '{node.id}' (op={node.op}), inputs={[t.shape for t in inputs]}: {str(e)}") from e
        return values[self.output_id]