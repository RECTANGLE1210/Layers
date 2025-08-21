import torch
from torch import nn
from typing import Optional
from units import ParamVec

class LinearRegressionBlock(nn.Module):
    """
    Linear Regression Block with ParamVec weights and bias.
    Can be used as a node in GraphLink.
    y = x @ W^T + b
    """

    def __init__(
        self,
        in_features: Optional[int] = None,
        out_features: int = 1,
        weight_init: str = "xavier_normal",
        bias_init: str = "zeros",
        weight_init_kwargs: Optional[dict] = None,
        bias_init_kwargs: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_shape = None
        self.output_shape = None
        self.input_shape_feat = None
        self.output_shape_feat = None

        if in_features is not None:
            self.weight = ParamVec(
                shape=(out_features, in_features),
                init=weight_init,
                init_kwargs=weight_init_kwargs,
                name=f"{name}_weight" if name else "linear_weight",
            )
            self.bias = ParamVec(
                shape=(out_features,),
                init=bias_init,
                init_kwargs=bias_init_kwargs,
                name=f"{name}_bias" if name else "linear_bias",
            )
        else:
            self.weight = None
            self.bias = None

        self.weight_init = weight_init
        self.bias_init = bias_init
        self.weight_init_kwargs = weight_init_kwargs
        self.bias_init_kwargs = bias_init_kwargs
        self.name = name

    def forward(self, inputs) -> torch.Tensor:
        """
        Forward pass for Linear Regression:
        Accepts either a tensor or list of tensors (as GraphLink provides).
        y = x @ W^T + b
        """
        if isinstance(inputs, list):
            x = inputs[0]  # lấy tensor đầu tiên
        else:
            x = inputs

        if self.weight is None or self.bias is None:
            in_features = x.shape[-1]
            self.in_features = in_features
            self.weight = ParamVec(
                shape=(self.out_features, in_features),
                init=self.weight_init,
                init_kwargs=self.weight_init_kwargs,
                name=f"{self.name}_weight" if self.name else "linear_weight",
            )
            self.bias = ParamVec(
                shape=(self.out_features,),
                init=self.bias_init,
                init_kwargs=self.bias_init_kwargs,
                name=f"{self.name}_bias" if self.name else "linear_bias",
            )
        else:
            if x.shape[-1] != self.in_features:
                raise ValueError(f"Expected input dimension {self.in_features}, got {x.shape[-1]}")

        result = torch.matmul(x, self.weight.tensor().t()) + self.bias.tensor()
        self.input_shape = tuple(x.shape)
        self.output_shape = tuple(result.shape)
        self.input_shape_feat = (x.shape[-1],)
        self.output_shape_feat = (result.shape[-1],)
        return result

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"

class LogisticRegressionBlock(nn.Module):
    """
    Logistic Regression Block with ParamVec weights and bias.
    Applies sigmoid activation to the output.
    y = sigmoid(x @ W^T + b)
    """
    def __init__(
        self,
        in_features: Optional[int] = None,
        out_features: int = 1,
        weight_init: str = "xavier_normal",
        bias_init: str = "zeros",
        weight_init_kwargs: Optional[dict] = None,
        bias_init_kwargs: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_shape = None
        self.output_shape = None
        self.input_shape_feat = None
        self.output_shape_feat = None

        if in_features is not None:
            self.weight = ParamVec(
                shape=(out_features, in_features),
                init=weight_init,
                init_kwargs=weight_init_kwargs,
                name=f"{name}_weight" if name else "logistic_weight",
            )
            self.bias = ParamVec(
                shape=(out_features,),
                init=bias_init,
                init_kwargs=bias_init_kwargs,
                name=f"{name}_bias" if name else "logistic_bias",
            )
        else:
            self.weight = None
            self.bias = None

        self.weight_init = weight_init
        self.bias_init = bias_init
        self.weight_init_kwargs = weight_init_kwargs
        self.bias_init_kwargs = bias_init_kwargs
        self.name = name

    def forward(self, inputs) -> torch.Tensor:
        """
        Forward pass for Logistic Regression:
        Accepts either a tensor or list of tensors.
        y = sigmoid(x @ W^T + b)
        """
        if isinstance(inputs, list):
            x = inputs[0]
        else:
            x = inputs

        if self.weight is None or self.bias is None:
            in_features = x.shape[-1]
            self.in_features = in_features
            self.weight = ParamVec(
                shape=(self.out_features, in_features),
                init=self.weight_init,
                init_kwargs=self.weight_init_kwargs,
                name=f"{self.name}_weight" if self.name else "logistic_weight",
            )
            self.bias = ParamVec(
                shape=(self.out_features,),
                init=self.bias_init,
                init_kwargs=self.bias_init_kwargs,
                name=f"{self.name}_bias" if self.name else "logistic_bias",
            )
        else:
            if x.shape[-1] != self.in_features:
                raise ValueError(f"Expected input dimension {self.in_features}, got {x.shape[-1]}")

        logits = torch.matmul(x, self.weight.tensor().t()) + self.bias.tensor()
        result = torch.sigmoid(logits)
        self.input_shape = tuple(x.shape)
        self.output_shape = tuple(result.shape)
        self.input_shape_feat = (x.shape[-1],)
        self.output_shape_feat = (result.shape[-1],)
        return result

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"

class SVMBlock(nn.Module):
    """
    Support Vector Machine Block with kernel support.
    - mode="SVC": Classification, output sign(score)
    - mode="SVR": Regression, output score
    Supports kernel_type: "linear", "poly", "rbf".
    """
    def __init__(
        self,
        in_features: Optional[int] = None,
        out_features: int = 1,
        mode: str = "SVC",
        kernel_type: str = "linear",
        degree: int = 3,
        gamma: Optional[float] = None,
        weight_init: str = "xavier_normal",
        bias_init: str = "zeros",
        name: Optional[str] = None,
    ):
        super().__init__()
        if mode not in ("SVC", "SVR"):
            raise ValueError(f"Unsupported mode: {mode}. Must be 'SVC' or 'SVR'.")
        if kernel_type not in ("linear", "poly", "rbf"):
            raise ValueError(f"Unsupported kernel_type: {kernel_type}. Must be 'linear', 'poly', or 'rbf'.")
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.kernel_type = kernel_type
        self.degree = degree
        self.gamma = gamma
        self.input_shape = None
        self.output_shape = None
        self.input_shape_feat = None
        self.output_shape_feat = None

        if in_features is not None:
            self.weight = ParamVec(
                shape=(out_features, in_features),
                init=weight_init,
                init_kwargs=None,
                name=f"{name}_weight" if name else "svm_weight",
            )
            self.bias = ParamVec(
                shape=(out_features,),
                init=bias_init,
                init_kwargs=None,
                name=f"{name}_bias" if name else "svm_bias",
            )
        else:
            self.weight = None
            self.bias = None

        self.weight_init = weight_init
        self.bias_init = bias_init
        self.name = name

    def compute_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel matrix between x1 and x2 according to self.kernel_type.
        x1: shape (N, in_features)
        x2: shape (M, in_features)
        Returns: shape (N, M)
        """
        if x1.shape[-1] != self.in_features or x2.shape[-1] != self.in_features:
            raise ValueError(f"Input features must match in_features={self.in_features}, got {x1.shape[-1]}, {x2.shape[-1]}")
        if self.kernel_type == "linear":
            return torch.matmul(x1, x2.t())
        elif self.kernel_type == "poly":
            gamma = self.gamma if self.gamma is not None else 1.0 / self.in_features
            return (gamma * torch.matmul(x1, x2.t()) + 1.0) ** self.degree
        elif self.kernel_type == "rbf":
            gamma = self.gamma if self.gamma is not None else 1.0 / self.in_features
            x1_sq = (x1 ** 2).sum(dim=1, keepdim=True)  # (N, 1)
            x2_sq = (x2 ** 2).sum(dim=1, keepdim=True)  # (M, 1)
            dist_sq = x1_sq - 2 * torch.matmul(x1, x2.t()) + x2_sq.t()  # (N, M)
            return torch.exp(-gamma * dist_sq)
        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type}")

    def forward(self, inputs) -> torch.Tensor:
        """
        Forward pass for SVM with kernel support.
        Accepts a tensor or list of tensors.
        - SVC: output = sign(score)
        - SVR: output = score
        """
        if isinstance(inputs, list):
            x = inputs[0]
        else:
            x = inputs

        if self.weight is None or self.bias is None:
            in_features = x.shape[-1]
            self.in_features = in_features
            self.weight = ParamVec(
                shape=(self.out_features, in_features),
                init=self.weight_init,
                init_kwargs=None,
                name=f"{self.name}_weight" if self.name else "svm_weight",
            )
            self.bias = ParamVec(
                shape=(self.out_features,),
                init=self.bias_init,
                init_kwargs=None,
                name=f"{self.name}_bias" if self.name else "svm_bias",
            )
        else:
            if x.shape[-1] != self.in_features:
                raise ValueError(f"Expected input dimension {self.in_features}, got {x.shape[-1]}")

        W = self.weight.tensor()  # (out_features, in_features)
        b = self.bias.tensor()    # (out_features,)
        K = self.compute_kernel(x, W)  # (N, out_features)
        score = K + b
        if self.mode == "SVC":
            result = torch.sign(score)
        else:  # SVR
            result = score
        self.input_shape = tuple(x.shape)
        self.output_shape = tuple(result.shape)
        self.input_shape_feat = (x.shape[-1],)
        self.output_shape_feat = (result.shape[-1],)
        return result

    def regularization(self, norm: str = "l2") -> torch.Tensor:
        """
        Returns the regularization term for weights.
        Currently only supports L2 norm.
        """
        if norm == "l2":
            return torch.norm(self.weight.tensor(), p=2)
        else:
            raise ValueError(f"Unsupported norm: {norm}")

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"mode={self.mode}, kernel_type={self.kernel_type}"
        )