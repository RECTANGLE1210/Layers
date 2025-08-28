import unittest
import torch
import torch.nn as nn
from graph_link import Node, GraphLink

# =========================
# Node wrappers cho CNN
# =========================
class Conv2dNode(Node):
    def __init__(self, id, inputs, conv_module):
        super().__init__(id=id, op="custom_block", inputs=inputs, kwargs={"fn": conv_module})

    def forward(self, input_tensors, x=None):
        return self.kwargs["fn"](input_tensors[0])

class FlattenNode(Node):
    def __init__(self, id, inputs):
        super().__init__(id=id, op="flatten", inputs=inputs, kwargs={})

    def forward(self, input_tensors, x=None):
        return input_tensors[0].flatten(start_dim=1)

class ActNode(Node):
    def __init__(self, id, inputs, act_type="relu"):
        super().__init__(id=id, op="act", inputs=inputs, kwargs={"act_type": act_type})

    def forward(self, input_tensors, x=None):
        t = input_tensors[0]
        if self.kwargs.get("act_type","relu").lower()=="relu":
            return torch.relu(t)
        else:
            raise ValueError(f"Unsupported act_type {self.kwargs.get('act_type')}")

# =========================
# Test case CNN minimal
# =========================
class TestGraphLinkCNN(unittest.TestCase):
    def test_cnn_minimal(self):
        batch_size = 4
        in_channels = 1
        H, W = 28, 28
        out_channels = 8
        kernel_size = 3

        # Conv module
        conv_module = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

        nodes = [
            Node(id="input", op="input", inputs=[], kwargs={"input_shape": (batch_size, in_channels, H, W)}),
            Conv2dNode(id="conv1", inputs=["input"], conv_module=conv_module),
            ActNode(id="relu1", inputs=["conv1"], act_type="relu"),
            FlattenNode(id="flatten", inputs=["relu1"]),
            Node(id="output", op="output_marker", inputs=["flatten"], kwargs={}),
        ]

        graph = GraphLink(nodes, output_id="output", trace=True)
        x = torch.randn(batch_size, in_channels, H, W)
        out = graph(x)

        # Kiểm tra shape output
        expected_dim = out_channels * (H - kernel_size + 1) * (W - kernel_size + 1)
        self.assertEqual(out.shape, (batch_size, expected_dim))
        print("CNN minimal test passed. Output shape:", out.shape)


if __name__=="__main__":
    unittest.main()