import torch
from torch import nn
import torch.nn.init as init
from typing import Optional, Dict, Any, Sequence, Union, Tuple

class ParamVec(nn.Module):
    """A learnable parameter tensor (vector or multi-dimensional) with lazy initialization support.

    Args:
        shape: The shape of the tensor. If int, treated as (int,). If None, will be lazily materialized.
        init: Initialization method. Supported: "zeros", "ones", "normal", "uniform",
            "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal".
        init_kwargs: Optional kwargs for initialization (e.g., {"mean": 0.0, "std": 0.02} for "normal").
            For fan-based inits like xavier, if "fan_out" is provided (int), it will assume the tensor is a flattened
            matrix of shape (fan_out, prod(shape)//fan_out) for proper fan_in/fan_out calculation.
        dtype: Data type of the parameter.
        device: Device for the parameter.
        name: Logical name for debugging.
        dropout_rate: Meta attribute for dropout rate (stored but not applied).
        freeze: Meta attribute for freeze (stored but not applied).

    Attributes:
        param: The underlying nn.Parameter (None if lazy).
        shape: Current shape (None if lazy).
        meta: Dictionary containing meta attributes.
    """

    def __init__(
        self,
        shape: Optional[Union[int, Tuple[int, ...]]] = None,
        init: str = "xavier_normal",
        init_kwargs: Optional[Dict[str, Any]] = None,
        dtype: torch.dtype = torch.float32,
        device: Union[torch.device, str, None] = None,
        name: Optional[str] = None,
        dropout_rate: Optional[float] = None,
        freeze: Optional[bool] = None,
    ):
        super().__init__()
        # Convert shape argument to tuple if int
        if isinstance(shape, int):
            shape_tuple = (shape,)
        elif shape is not None:
            shape_tuple = tuple(shape)
        else:
            shape_tuple = None
        self._shape: Optional[Tuple[int, ...]] = shape_tuple
        self._init: str = init
        self._init_kwargs: Dict[str, Any] = init_kwargs or {}
        self._dtype: torch.dtype = dtype
        self._device: torch.device = torch.device(device) if isinstance(device, str) else (device or torch.device("cpu"))
        self._param: Optional[nn.Parameter] = None
        self.meta: Dict[str, Any] = {
            "name": name,
            "dropout_rate": dropout_rate,
            "freeze": freeze,
        }
        if shape_tuple is not None:
            self._materialize(shape_tuple)

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        return self._shape

    @property
    def param(self) -> Optional[nn.Parameter]:
        return self._param

    def _materialize(self, shape: Tuple[int, ...]) -> None:
        if self._shape is not None and self._shape != shape:
            raise ValueError(f"Cannot materialize with shape {shape}; already set to {self._shape}")
        tensor = torch.empty(*shape, dtype=self._dtype, device=self._device)
        numel = int(torch.tensor(shape).prod().item()) if shape else 1
        if self._init in ["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"]:
            # For fan-based inits, try to reshape to (fan_out, prod//fan_out) if fan_out is provided
            if "fan_out" in self._init_kwargs:
                fan_out = int(self._init_kwargs["fan_out"])
                prod = numel
                fan_in = prod // fan_out
                if fan_in * fan_out != prod:
                    raise ValueError(f"Shape {shape} (numel={prod}) not divisible by fan_out {fan_out}")
                temp = torch.empty(fan_out, fan_in, dtype=self._dtype, device=self._device)
            else:
                temp = tensor.unsqueeze(0)  # (1, ...)
            if self._init == "xavier_uniform":
                gain = self._init_kwargs.get("gain", 1.0)
                init.xavier_uniform_(temp, gain=gain)
            elif self._init == "xavier_normal":
                gain = self._init_kwargs.get("gain", 1.0)
                init.xavier_normal_(temp, gain=gain)
            elif self._init == "kaiming_uniform":
                a = self._init_kwargs.get("a", 0)
                mode = self._init_kwargs.get("mode", "fan_in")
                nonlinearity = self._init_kwargs.get("nonlinearity", "leaky_relu")
                init.kaiming_uniform_(temp, a=a, mode=mode, nonlinearity=nonlinearity)
            elif self._init == "kaiming_normal":
                a = self._init_kwargs.get("a", 0)
                mode = self._init_kwargs.get("mode", "fan_in")
                nonlinearity = self._init_kwargs.get("nonlinearity", "leaky_relu")
                init.kaiming_normal_(temp, a=a, mode=mode, nonlinearity=nonlinearity)
            if "fan_out" in self._init_kwargs:
                tensor = temp.view(*shape)
            else:
                tensor = temp.squeeze(0)
        else:
            if self._init == "zeros":
                tensor.zero_()
            elif self._init == "ones":
                tensor.fill_(1.0)
            elif self._init == "normal":
                mean = self._init_kwargs.get("mean", 0.0)
                std = self._init_kwargs.get("std", 1.0)
                tensor.normal_(mean, std)
            elif self._init == "uniform":
                a = self._init_kwargs.get("a", 0.0)
                b = self._init_kwargs.get("b", 1.0)
                tensor.uniform_(a, b)
            else:
                raise ValueError(f"Unsupported init: {self._init}")
        self._param = nn.Parameter(tensor, requires_grad=True)
        self._shape = shape
        self.register_parameter("_param", self._param)

    def tensor(self) -> torch.Tensor:
        if self._param is None:
            raise ValueError("ParamVec is lazy and not materialized yet")
        return self._param

    def set_shape(self, shape: Union[int, Tuple[int, ...]]) -> None:
        if isinstance(shape, int):
            shape_tuple = (shape,)
        else:
            shape_tuple = tuple(shape)
        self._materialize(shape_tuple)

    def _ensure_materialized(self, target_shape: Optional[Union[int, Tuple[int, ...]]] = None) -> None:
        if self._shape is None:
            if target_shape is None:
                raise ValueError("Cannot materialize without a target shape")
            if isinstance(target_shape, int):
                target_shape_tuple = (target_shape,)
            else:
                target_shape_tuple = tuple(target_shape)
            self._materialize(target_shape_tuple)

    def __add__(self, other: "ParamVec") -> torch.Tensor:
        # Materialize to compatible shapes if needed
        s_shape = self.shape
        o_shape = other.shape
        # If both lazy, error
        if s_shape is None and o_shape is None:
            raise ValueError("Both ParamVec are lazy; cannot determine shape")
        known_shape = s_shape if s_shape is not None else o_shape
        self._ensure_materialized(known_shape)
        other._ensure_materialized(known_shape)
        return self.tensor() + other.tensor()

    def __sub__(self, other: "ParamVec") -> torch.Tensor:
        s_shape = self.shape
        o_shape = other.shape
        if s_shape is None and o_shape is None:
            raise ValueError("Both ParamVec are lazy; cannot determine shape")
        known_shape = s_shape if s_shape is not None else o_shape
        self._ensure_materialized(known_shape)
        other._ensure_materialized(known_shape)
        return self.tensor() - other.tensor()

    def __mul__(self, other: "ParamVec") -> torch.Tensor:
        s_shape = self.shape
        o_shape = other.shape
        if s_shape is None and o_shape is None:
            raise ValueError("Both ParamVec are lazy; cannot determine shape")
        known_shape = s_shape if s_shape is not None else o_shape
        self._ensure_materialized(known_shape)
        other._ensure_materialized(known_shape)
        return self.tensor() * other.tensor()

    def __matmul__(self, other: "ParamVec") -> torch.Tensor:
        s_shape = self.shape
        o_shape = other.shape
        if s_shape is None and o_shape is None:
            raise ValueError("Both ParamVec are lazy; cannot determine shape for matmul")
        known_shape = s_shape if s_shape is not None else o_shape
        self._ensure_materialized(known_shape)
        other._ensure_materialized(known_shape)
        s_tensor = self.tensor()
        o_tensor = other.tensor()
        # If both are 1D, use torch.dot, else use torch.matmul (broadcasts as needed)
        if s_tensor.dim() == 1 and o_tensor.dim() == 1:
            return torch.dot(s_tensor, o_tensor)
        else:
            return torch.matmul(s_tensor, o_tensor)

    def sin(self) -> torch.Tensor:
        self._ensure_materialized()
        return torch.sin(self.tensor())

    def cos(self) -> torch.Tensor:
        self._ensure_materialized()
        return torch.cos(self.tensor())

    def tan(self) -> torch.Tensor:
        self._ensure_materialized()
        return torch.tan(self.tensor())

    def tanh(self) -> torch.Tensor:
        self._ensure_materialized()
        return torch.tanh(self.tensor())

    def relu(self) -> torch.Tensor:
        self._ensure_materialized()
        return torch.relu(self.tensor())

    def softmax(self, dim: Optional[int] = None) -> torch.Tensor:
        self._ensure_materialized()
        # Default to softmax along first dim if not specified
        if dim is None:
            dim = 0
        return torch.softmax(self.tensor(), dim=dim)

    def to(self, *args, **kwargs) -> "ParamVec":
        super().to(*args, **kwargs)
        self._device = self._param.device if self._param is not None else self._device
        self._dtype = self._param.dtype if self._param is not None else self._dtype
        return self

    def cpu(self) -> "ParamVec":
        return self.to("cpu")

    def cuda(self) -> "ParamVec":
        return self.to("cuda")

    def detach_copy(self) -> torch.Tensor:
        return self.tensor().detach().clone()

    def extra_repr(self) -> str:
        return f"shape={self.shape}, init={self._init}, meta={self.meta}"

    def __repr__(self) -> str:
        return f"ParamVec({self.extra_repr()})"

def concat(vs: Sequence[ParamVec], *, dim_policy: str = "auto") -> torch.Tensor:
    """Concatenate a sequence of ParamVec along dim=0.

    Args:
        vs: Sequence of ParamVec to concatenate.
        dim_policy: "strict" requires all shapes known and first dimension equal; "auto" materializes lazies to common known shape.

    Returns:
        Concatenated tensor.
    """
    if not vs:
        raise ValueError("Cannot concat empty sequence")
    known_shapes = [v.shape for v in vs if v.shape is not None]
    if not known_shapes:
        raise ValueError("No known shapes among ParamVecs for concat")
    # All shapes must have the same first dimension for concat
    first_dims = {s[0] for s in known_shapes}
    if len(first_dims) > 1:
        raise ValueError(f"Multiple different first dims for concat: {first_dims}")
    common_first_dim = first_dims.pop()
    # For strict: all shapes must be known and match
    if dim_policy == "strict":
        if any(v.shape is None for v in vs):
            raise ValueError("Strict policy requires all shapes known")
        if any(v.shape[0] != common_first_dim for v in vs):
            raise ValueError("Strict policy requires all first dims equal")
    elif dim_policy == "auto":
        for v in vs:
            if v.shape is None:
                # Materialize to shape (common_first_dim,)
                v._ensure_materialized((common_first_dim,))
            elif v.shape[0] != common_first_dim:
                raise ValueError("Auto policy requires all first dims to match")
    else:
        raise ValueError(f"Unknown dim_policy: {dim_policy}")
    tensors = [v.tensor() for v in vs]
    return torch.cat(tensors, dim=0)


def dot(a: ParamVec, b: ParamVec) -> torch.Tensor:
    """Dot product convenience function."""
    return a @ b

"""
------------------------------------------------------------------------------
units.py - ParamVec, GraphLink: Modular Parameter Vectors and Composable Neural Computation Graphs
------------------------------------------------------------------------------

This file provides a flexible framework for constructing neural network computation graphs using modular, learnable parameter vectors and composable operation chains. It is designed for research, experimentation, and rapid prototyping of custom neural architectures.

Core Components:
----------------

1. ParamVec
-----------
ParamVec is a learnable parameter vector (subclass of nn.Module) that supports lazy initialization, flexible initialization schemes, and convenient meta attributes.

- **Purpose**: Encapsulates a single trainable parameter vector, optionally with a logical name and meta attributes (like dropout rate, freeze flag).
- **Lazy Initialization**: If the dimension (`dim`) is not provided, the vector is "lazy" and will be materialized (allocated and initialized) only when its size is known (e.g., upon first operation or explicit set_dim).
- **Supported Initializations**: "zeros", "ones", "normal", "uniform", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal". Some initializations (e.g., fan-based) accept additional kwargs (e.g., `fan_out`, `gain`, etc.).
- **Broadcasting**: ParamVec supports elementwise binary ops (+, -, *, dot) with broadcasting when one side is scalar (dim=1).
- **Materialization**: The vector can be materialized to a specific dimension via `.set_dim()` or automatically during operations.
- **Convenience methods**: Activation functions (sin, cos, tan, tanh, relu, softmax), device/dtype transfer, and detach/copy.

2. concat and dot
-----------------
These are utility functions for combining and operating on ParamVec instances:

- **concat**: Concatenates a sequence of ParamVecs along dimension 0. Supports "strict" (all dims known and equal) or "auto" (materialize lazies to common dim) policies.
- **dot**: Computes the dot product between two ParamVecs, handling lazy initialization and broadcasting as needed.

3. GraphLink
------------
GraphLink enables the definition and execution of computation graphs (directed acyclic graphs, DAGs) with arbitrary node operations.

- **Purpose**: Provides a general DAG abstraction for neural computation, where each node specifies an operation, inputs (by node id), and kwargs.
- **Graph Definition**:
    - Nodes are dicts with keys: "id", "op", "inputs", "kwargs".
    - Supported node ops: "input", "input_marker", "output_marker", "linear", "add", "mul", "dot", "concat", "act", "custom".
    - The output node is specified by `output_id`.
- **Operation Execution**:
    - Topological sort is performed to determine evaluation order.
    - Each node pulls its inputs from previous node outputs and executes its operation.
    - Lazy ParamVecs are materialized as needed to match input shapes.
- **Trace Mode**: If enabled, prints node-by-node trace of computation and shapes.
"""