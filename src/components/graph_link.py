import torch
from torch import nn
from typing import Dict, Any, List

class GraphLink(nn.Module):
    """A module for defining a computation graph (DAG) with ParamVec operations.

    Args:
        nodes: List of node dicts with keys: id (str), op (str), inputs (List[str]), kwargs (dict).
        output_id: ID of the output node.
        trace: If True, print trace during forward.
    """

    def __init__(self, nodes: List[Dict[str, Any]], output_id: str, trace: bool = False):
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
        """Compute topological order of nodes."""
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
        values = {"input": x}
        for nid in self.topo_order:
            node = self.nodes[nid]
            op = node["op"]
            inputs = [values[inp] for inp in node["inputs"]]
            kwargs = node["kwargs"]
            if self.trace:
                print(f"Node {nid}: op={op}, input shapes={[t.shape for t in inputs]}")
            if op == "input":
                if nid != "input":
                    raise ValueError("Only one node can have op='input' with id='input'")
                values[nid] = x
            elif op == "input_marker":
                values[nid] = x
            elif op == "output_marker":
                values[nid] = inputs[0] if inputs else x
            elif op == "linear":
                dim_in = inputs[0].shape[-1] if inputs[0].dim() >= 1 else 1
                dim_out = kwargs["dim_out"]
                weight = kwargs["weight"]
                weight._ensure_materialized((dim_out, dim_in))
                w_flat = weight.tensor()
                w_t = w_flat.view(dim_out, dim_in)
                result = torch.matmul(inputs[0], w_t.T)
                if "bias" in kwargs and kwargs["bias"] is not None:
                    bias = kwargs["bias"]
                    bias._ensure_materialized((dim_out,))
                    b_t = bias.tensor()
                    if getattr(bias, "shape", None) is not None and len(bias.shape) == 1 and bias.shape[0] == 1:
                        b_t = b_t.expand_as(result)
                    result += b_t
                values[nid] = result
            elif op == "add":
                if "param" in kwargs:
                    param = kwargs["param"]
                    dim = inputs[0].shape[-1] if inputs[0].dim() >= 1 else 1
                    param._ensure_materialized((dim,))
                    p_t = param.tensor()
                    if getattr(param, "shape", None) is not None and len(param.shape) == 1 and param.shape[0] == 1:
                        p_t = p_t.expand_as(inputs[0])
                    values[nid] = inputs[0] + p_t
                else:
                    # Residual-style: sum all inputs
                    result = inputs[0]
                    for inp in inputs[1:]:
                        result = result + inp
                    values[nid] = result
            elif op == "mul":
                if "param" in kwargs:
                    param = kwargs["param"]
                    dim = inputs[0].shape[-1] if inputs[0].dim() >= 1 else 1
                    param._ensure_materialized((dim,))
                    p_t = param.tensor()
                    if getattr(param, "shape", None) is not None and len(param.shape) == 1 and param.shape[0] == 1:
                        p_t = p_t.expand_as(inputs[0])
                    values[nid] = inputs[0] * p_t
                else:
                    # Multiply all inputs together
                    result = inputs[0]
                    for inp in inputs[1:]:
                        result = result * inp
                    values[nid] = result
            elif op == "dot":
                param = kwargs["param"]
                dim = inputs[0].shape[-1] if inputs[0].dim() >= 1 else 1
                param._ensure_materialized((dim,))
                p_t = param.tensor()
                p_dim = p_t.shape[0] if p_t.dim() >= 1 else 1
                if dim == p_dim:
                    values[nid] = torch.sum(inputs[0] * p_t, dim=-1)
                elif p_dim == 1:
                    values[nid] = p_t * torch.sum(inputs[0], dim=-1)
                elif dim == 1:
                    values[nid] = torch.sum(p_t) * inputs[0].squeeze(-1) if inputs[0].dim() >= 1 else torch.sum(p_t) * inputs[0]
                else:
                    raise ValueError("Shape mismatch for dot")
            elif op == "concat":
                # Concat outputs of previous nodes (inputs), not ParamVec weights
                values[nid] = torch.cat(inputs, dim=-1)
            elif op == "act":
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
                fn = kwargs["fn"]
                values[nid] = fn(inputs)
            else:
                raise ValueError(f"Unsupported op: {op}")
        return values[self.output_id]