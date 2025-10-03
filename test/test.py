import unittest
import torch

from components.graph_link import Node, GraphLink
from components.units import ParamVec

class TestGraphLink(unittest.TestCase):

    def test_dag_no_shape(self):
        # User does not specify shape (automatic selection)
        weight1 = ParamVec(shape=None, init="xavier_uniform", init_kwargs={"fan_out": 4})
        bias1 = ParamVec(shape=None, init="zeros")
        weight2 = ParamVec(shape=None, init="xavier_uniform", init_kwargs={"fan_out": 4})
        bias2 = ParamVec(shape=None, init="zeros")

        batch_size = 8
        input_dim = 1

        nodes = [
            Node(id="input", op="input", inputs=[], kwargs={"input_shape": (batch_size, input_dim)}),
            Node(id="linear1", op="linear", inputs=["input"], kwargs={"dim_out": 4, "weight": weight1, "bias": bias1}),
            Node(id="linear2", op="linear", inputs=["input"], kwargs={"dim_out": 4, "weight": weight2, "bias": bias2}),
            Node(id="concat", op="concat", inputs=["linear1", "linear2"], kwargs={}),
            Node(id="softmax", op="act", inputs=["concat"], kwargs={"act_type": "softmax"}),
        ]
        graph = GraphLink(nodes, output_id="softmax", trace=True)

        x = torch.randn(batch_size, input_dim)  # batch_size=8, input_dim=1
        out = graph(x)
        self.assertEqual(out.shape, (batch_size, 8))  # concat 2*4 output => shape (8,8)

    def test_dag_correct_shape(self):
        # User specifies correct shape
        weight1 = ParamVec(shape=(4, 8), init="xavier_uniform", init_kwargs={"fan_out": 4})
        bias1 = ParamVec(shape=(4,), init="zeros")
        weight2 = ParamVec(shape=(4, 8), init="xavier_uniform", init_kwargs={"fan_out": 4})
        bias2 = ParamVec(shape=(4,), init="zeros")

        batch_size = 8
        input_dim = 8

        nodes = [
            Node(id="input", op="input", inputs=[], kwargs={"input_shape": (batch_size, input_dim)}),
            Node(id="linear1", op="linear", inputs=["input"], kwargs={"dim_out": 4, "weight": weight1, "bias": bias1}),
            Node(id="linear2", op="linear", inputs=["input"], kwargs={"dim_out": 4, "weight": weight2, "bias": bias2}),
            Node(id="concat", op="concat", inputs=["linear1", "linear2"], kwargs={}),
            Node(id="softmax", op="act", inputs=["concat"], kwargs={"act_type": "softmax"}),
        ]
        graph = GraphLink(nodes, output_id="softmax", trace=True)

        x = torch.randn(batch_size, input_dim)
        out = graph(x)
        self.assertEqual(out.shape, (batch_size, 8))

    def test_dag_incorrect_shape(self):
        # User specifies incorrect shape (should raise exception)
        weight1 = ParamVec(shape=(4, 5), init="xavier_uniform", init_kwargs={"fan_out": 4})
        bias1 = ParamVec(shape=(4,), init="zeros")
        weight2 = ParamVec(shape=(4, 8), init="xavier_uniform", init_kwargs={"fan_out": 4})
        bias2 = ParamVec(shape=(4,), init="zeros")

        batch_size = 8
        input_dim = 8

        nodes = [
            Node(id="input", op="input", inputs=[], kwargs={"input_shape": (batch_size, input_dim)}),
            Node(id="linear1", op="linear", inputs=["input"], kwargs={"dim_out": 4, "weight": weight1, "bias": bias1}),
            Node(id="linear2", op="linear", inputs=["input"], kwargs={"dim_out": 4, "weight": weight2, "bias": bias2}),
            Node(id="concat", op="concat", inputs=["linear1", "linear2"], kwargs={}),
            Node(id="softmax", op="act", inputs=["concat"], kwargs={"act_type": "softmax"}),
        ]
        graph = GraphLink(nodes, output_id="softmax", trace=True)

        x = torch.randn(batch_size, input_dim)
        with self.assertRaises(RuntimeError) as context:
            out = graph(x)

        expected_error = "[GraphLink] Shape mismatch at node 'linear1' (op=linear), inputs=[torch.Size([8, 8])]: [GraphLink] Error in node 'linear1' (op=linear): Invalid weight shape (4, 5) for node linear1, possible shapes: [(4, 8)]"
        self.assertEqual(str(context.exception), expected_error)

    def test_residual_block_no_shape(self):
        input_dim = 15
        batch_size = 32
        # User does not specify shape (automatic selection)
        weight1 = ParamVec(shape=None, init="xavier_uniform", init_kwargs={"fan_out": input_dim})
        bias1 = ParamVec(shape=None, init="zeros")
        weight2 = ParamVec(shape=None, init="xavier_uniform", init_kwargs={"fan_out": input_dim})
        bias2 = ParamVec(shape=None, init="zeros")

        nodes = [
            Node(id="input", op="input", inputs=[], kwargs={"input_shape": (batch_size, input_dim)}),
            Node(id="linear1", op="linear", inputs=["input"], kwargs={"dim_out": input_dim, "weight": weight1, "bias": bias1}),
            Node(id="relu", op="act", inputs=["linear1"], kwargs={"act_type": "relu"}),
            Node(id="linear2", op="linear", inputs=["relu"], kwargs={"dim_out": input_dim, "weight": weight2, "bias": bias2}),
            Node(id="add", op="add", inputs=["linear2", "input"], kwargs={}),
            Node(id="output", op="output_marker", inputs=["add"], kwargs={}),
        ]

        graph = GraphLink(nodes, output_id="output", trace=True)
        x = torch.randn(batch_size, input_dim)
        out = graph(x)
        self.assertEqual(out.shape, (batch_size, input_dim))

    def test_residual_block_correct_shape(self):
        input_dim = 15
        batch_size = 32
        # User specifies correct shape
        weight1 = ParamVec(shape=(input_dim, input_dim), init="xavier_uniform", init_kwargs={"fan_out": input_dim})
        bias1 = ParamVec(shape=(input_dim,), init="zeros")
        weight2 = ParamVec(shape=(input_dim, input_dim), init="xavier_uniform", init_kwargs={"fan_out": input_dim})
        bias2 = ParamVec(shape=(input_dim,), init="zeros")

        nodes = [
            Node(id="input", op="input", inputs=[], kwargs={"input_shape": (batch_size, input_dim)}),
            Node(id="linear1", op="linear", inputs=["input"], kwargs={"dim_out": input_dim, "weight": weight1, "bias": bias1}),
            Node(id="relu", op="act", inputs=["linear1"], kwargs={"act_type": "relu"}),
            Node(id="linear2", op="linear", inputs=["relu"], kwargs={"dim_out": input_dim, "weight": weight2, "bias": bias2}),
            Node(id="add", op="add", inputs=["linear2", "input"], kwargs={}),
            Node(id="output", op="output_marker", inputs=["add"], kwargs={}),
        ]

        graph = GraphLink(nodes, output_id="output", trace=True)
        x = torch.randn(batch_size, input_dim)
        out = graph(x)
        self.assertEqual(out.shape, (batch_size, input_dim))

    def test_residual_block_incorrect_shape(self):
        input_dim = 15
        batch_size = 32
        # User specifies incorrect shape (should raise exception)
        weight1 = ParamVec(shape=(input_dim, 10), init="xavier_uniform", init_kwargs={"fan_out": input_dim})
        bias1 = ParamVec(shape=(input_dim,), init="zeros")
        weight2 = ParamVec(shape=(input_dim, input_dim), init="xavier_uniform", init_kwargs={"fan_out": input_dim})
        bias2 = ParamVec(shape=(input_dim,), init="zeros")

        nodes = [
            Node(id="input", op="input", inputs=[], kwargs={"input_shape": (batch_size, input_dim)}),
            Node(id="linear1", op="linear", inputs=["input"], kwargs={"dim_out": input_dim, "weight": weight1, "bias": bias1}),
            Node(id="relu", op="act", inputs=["linear1"], kwargs={"act_type": "relu"}),
            Node(id="linear2", op="linear", inputs=["relu"], kwargs={"dim_out": input_dim, "weight": weight2, "bias": bias2}),
            Node(id="add", op="add", inputs=["linear2", "input"], kwargs={}),
            Node(id="output", op="output_marker", inputs=["add"], kwargs={}),
        ]
        graph = GraphLink(nodes, output_id="output", trace=True)
        x = torch.randn(batch_size, input_dim)
        with self.assertRaises(RuntimeError) as context:
            out = graph(x)
        expected_error = f"[GraphLink] Shape mismatch at node 'linear1' (op=linear), inputs=[torch.Size([{batch_size}, {input_dim}])]: [GraphLink] Error in node 'linear1' (op=linear): Invalid weight shape ({input_dim}, 10) for node linear1, possible shapes: [({input_dim}, {input_dim})]"
        self.assertEqual(str(context.exception), expected_error)

if __name__ == "__main__":
    unittest.main()