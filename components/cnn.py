import torch
import torch.nn as nn
import torch.nn.functional as F
from units import ParamVec
from graph_link import Node, GraphLink

# =========================
# CNN Node Classes
# =========================

class Conv2dNode(Node):
    """
    Node wrapper cho 2D convolution.
    Có thể dùng ParamVec hoặc nn.Conv2d trực tiếp.
    """
    def __init__(self, id, inputs, weight=None, bias=None, conv_module=None, stride=1, padding=0):
        """
        weight, bias: ParamVec objects
        conv_module: nn.Conv2d object
        """
        kwargs = dict(weight=weight, bias=bias, conv_module=conv_module, stride=stride, padding=padding)
        super().__init__(id=id, op="conv2d", inputs=inputs, kwargs=kwargs)

    def forward(self, input_tensors, x=None):
        x = input_tensors[0]
        conv_module = self.kwargs.get("conv_module")
        if conv_module is not None:
            return conv_module(x)
        else:
            weight = self.kwargs["weight"].tensor
            bias = self.kwargs.get("bias")
            return F.conv2d(x, weight, bias=bias, stride=self.kwargs.get("stride",1), padding=self.kwargs.get("padding",0))


class FlattenNode(Node):
    """
    Node flatten: chuyển tensor nhiều chiều sang 2D (B, -1)
    """
    def __init__(self, id, inputs):
        super().__init__(id=id, op="flatten", inputs=inputs, kwargs={})

    def forward(self, input_tensors, x=None):
        return input_tensors[0].flatten(start_dim=1)


class ActNode(Node):
    """
    Activation node: relu, sigmoid, tanh
    """
    def __init__(self, id, inputs, act_type="relu"):
        super().__init__(id=id, op="act", inputs=inputs, kwargs={"act_type": act_type})

    def forward(self, input_tensors, x=None):
        t = input_tensors[0]
        act_type = self.kwargs.get("act_type","relu").lower()
        if act_type=="relu":
            return F.relu(t)
        elif act_type=="sigmoid":
            return torch.sigmoid(t)
        elif act_type=="tanh":
            return torch.tanh(t)
        else:
            raise ValueError(f"Unsupported act_type {act_type}")


# =========================
# Helper functions
# =========================

def build_basic_cnn(input_shape, conv_channels, kernel_size=(3,3), use_paramvec=True):
    """
    Tạo GraphLink cho CNN cơ bản:
    - input -> conv -> relu -> flatten
    input_shape: (C,H,W)
    conv_channels: int
    kernel_size: tuple
    use_paramvec: nếu True dùng ParamVec, nếu False dùng nn.Conv2d
    """
    nodes = []
    C,H,W = input_shape

    # Input node
    nodes.append(Node(id="input", op="input", inputs=[], kwargs={"input_shape": (1,C,H,W)}))

    # Conv node
    if use_paramvec:
        weight = ParamVec(shape=(conv_channels, C, *kernel_size), init="xavier_uniform")
        bias = ParamVec(shape=(conv_channels,), init="zeros")
        conv_node = Conv2dNode(id="conv1", inputs=["input"], weight=weight, bias=bias, stride=1, padding=0)
    else:
        conv_module = nn.Conv2d(in_channels=C, out_channels=conv_channels, kernel_size=kernel_size)
        conv_node = Conv2dNode(id="conv1", inputs=["input"], conv_module=conv_module)
    nodes.append(conv_node)

    # Activation
    nodes.append(ActNode(id="relu1", inputs=["conv1"], act_type="relu"))

    # Flatten
    nodes.append(FlattenNode(id="flatten", inputs=["relu1"]))

    # Build GraphLink
    graph = GraphLink(nodes, output_id="flatten", trace=True)
    return graph


# =========================
# Example usage
# =========================

if __name__=="__main__":
    # Minimal example
    x = torch.randn(1,1,28,28)  # batch_size=1
    graph = build_basic_cnn(input_shape=(1,28,28), conv_channels=8, use_paramvec=False)
    out = graph(x)
    print("Output shape:", out.shape)