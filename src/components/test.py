import unittest
import torch
from torch import nn
from typing import Dict, Any, List
from graph_link import GraphLink
from units import ParamVec
# Assuming ParamVec and GraphLink classes are defined as provided previously

class TestGraphLink(unittest.TestCase):

    def test_dag_no_shape(self):
        # User does not specify shape (automatic selection)
        weight1 = ParamVec(shape=None, init="xavier_uniform", init_kwargs={"fan_out": 4})
        bias1 = ParamVec(shape=None, init="zeros")
        weight2 = ParamVec(shape=None, init="xavier_uniform", init_kwargs={"fan_out": 4})
        bias2 = ParamVec(shape=None, init="zeros")
        nodes = [
            {"id": "input", "op": "input", "inputs": [], "kwargs": {}},
            {"id": "linear1", "op": "linear", "inputs": ["input"], "kwargs": {"dim_out": 4, "weight": weight1, "bias": bias1}},
            {"id": "linear2", "op": "linear", "inputs": ["input"], "kwargs": {"dim_out": 4, "weight": weight2, "bias": bias2}},
            {"id": "concat", "op": "concat", "inputs": ["linear1", "linear2"], "kwargs": {}},
            {"id": "softmax", "op": "act", "inputs": ["concat"], "kwargs": {"act_type": "softmax"}},
        ]
        graph = GraphLink(nodes, output_id="softmax", trace=True)
        x = torch.randn(8)
        out = graph(x)
        self.assertEqual(out.shape, (8,))

    def test_dag_correct_shape(self):
        # User specifies correct shape
        weight1 = ParamVec(shape=(4, 8), init="xavier_uniform", init_kwargs={"fan_out": 4})
        bias1 = ParamVec(shape=(4,), init="zeros")
        weight2 = ParamVec(shape=(4, 8), init="xavier_uniform", init_kwargs={"fan_out": 4})
        bias2 = ParamVec(shape=(4,), init="zeros")
        nodes = [
            {"id": "input", "op": "input", "inputs": [], "kwargs": {}},
            {"id": "linear1", "op": "linear", "inputs": ["input"], "kwargs": {"dim_out": 4, "weight": weight1, "bias": bias1}},
            {"id": "linear2", "op": "linear", "inputs": ["input"], "kwargs": {"dim_out": 4, "weight": weight2, "bias": bias2}},
            {"id": "concat", "op": "concat", "inputs": ["linear1", "linear2"], "kwargs": {}},
            {"id": "softmax", "op": "act", "inputs": ["concat"], "kwargs": {"act_type": "softmax"}},
        ]
        graph = GraphLink(nodes, output_id="softmax", trace=True)
        x = torch.randn(8)
        out = graph(x)
        self.assertEqual(out.shape, (8,))

    def test_dag_incorrect_shape(self):
        # User specifies incorrect shape (should raise exception)
        weight1 = ParamVec(shape=(4, 5), init="xavier_uniform", init_kwargs={"fan_out": 4})  # Wrong input dim (should be 8)
        bias1 = ParamVec(shape=(4,), init="zeros")  # Correct bias shape
        weight2 = ParamVec(shape=(4, 8), init="xavier_uniform", init_kwargs={"fan_out": 4})  # Correct weight shape
        bias2 = ParamVec(shape=(4,), init="zeros")
        nodes = [
            {"id": "input", "op": "input", "inputs": [], "kwargs": {}},
            {"id": "linear1", "op": "linear", "inputs": ["input"], "kwargs": {"dim_out": 4, "weight": weight1, "bias": bias1}},
            {"id": "linear2", "op": "linear", "inputs": ["input"], "kwargs": {"dim_out": 4, "weight": weight2, "bias": bias2}},
            {"id": "concat", "op": "concat", "inputs": ["linear1", "linear2"], "kwargs": {}},
            {"id": "softmax", "op": "act", "inputs": ["concat"], "kwargs": {"act_type": "softmax"}},
        ]
        graph = GraphLink(nodes, output_id="softmax", trace=True)
        x = torch.randn(8)  # Input dimension is 8
        with self.assertRaises(ValueError) as context:
            out = graph(x)
        expected_error = "Invalid weight shape (4, 5) for node linear1, possible shapes: [(4, 8)]"
        self.assertEqual(str(context.exception), expected_error, f"Expected error message '{expected_error}', got '{str(context.exception)}'")

    def test_residual_block_no_shape(self):
        input_dim = 15
        batch_size = 32
        # User does not specify shape (automatic selection)
        weight1 = ParamVec(shape=None, init="xavier_uniform", init_kwargs={"fan_out": input_dim})
        bias1 = ParamVec(shape=None, init="zeros")
        weight2 = ParamVec(shape=None, init="xavier_uniform", init_kwargs={"fan_out": input_dim})
        bias2 = ParamVec(shape=None, init="zeros")

        nodes = [
            {"id": "input", "op": "input", "inputs": [], "kwargs": {}},
            {"id": "linear1", "op": "linear", "inputs": ["input"], "kwargs": {"dim_out": input_dim, "weight": weight1, "bias": bias1}},
            {"id": "relu", "op": "act", "inputs": ["linear1"], "kwargs": {"act_type": "relu"}},
            {"id": "linear2", "op": "linear", "inputs": ["relu"], "kwargs": {"dim_out": input_dim, "weight": weight2, "bias": bias2}},
            {"id": "add", "op": "add", "inputs": ["linear2", "input"], "kwargs": {}},
            {"id": "output", "op": "output_marker", "inputs": ["add"], "kwargs": {}},
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
            {"id": "input", "op": "input", "inputs": [], "kwargs": {}},
            {"id": "linear1", "op": "linear", "inputs": ["input"], "kwargs": {"dim_out": input_dim, "weight": weight1, "bias": bias1}},
            {"id": "relu", "op": "act", "inputs": ["linear1"], "kwargs": {"act_type": "relu"}},
            {"id": "linear2", "op": "linear", "inputs": ["relu"], "kwargs": {"dim_out": input_dim, "weight": weight2, "bias": bias2}},
            {"id": "add", "op": "add", "inputs": ["linear2", "input"], "kwargs": {}},
            {"id": "output", "op": "output_marker", "inputs": ["add"], "kwargs": {}},
        ]

        graph = GraphLink(nodes, output_id="output", trace=True)
        x = torch.randn(batch_size, input_dim)
        out = graph(x)
        self.assertEqual(out.shape, (batch_size, input_dim))

    def test_residual_block_incorrect_shape(self):
        input_dim = 15
        batch_size = 32
        # User specifies incorrect shape (should raise exception)
        weight1 = ParamVec(shape=(input_dim, 10), init="xavier_uniform", init_kwargs={"fan_out": input_dim})  # Wrong input dim (should be 15)
        bias1 = ParamVec(shape=(input_dim,), init="zeros")  # Correct bias shape
        weight2 = ParamVec(shape=(input_dim, input_dim), init="xavier_uniform", init_kwargs={"fan_out": input_dim})  # Correct weight shape
        bias2 = ParamVec(shape=(input_dim,), init="zeros")
        nodes = [
            {"id": "input", "op": "input", "inputs": [], "kwargs": {}},
            {"id": "linear1", "op": "linear", "inputs": ["input"], "kwargs": {"dim_out": input_dim, "weight": weight1, "bias": bias1}},
            {"id": "relu", "op": "act", "inputs": ["linear1"], "kwargs": {"act_type": "relu"}},
            {"id": "linear2", "op": "linear", "inputs": ["relu"], "kwargs": {"dim_out": input_dim, "weight": weight2, "bias": bias2}},
            {"id": "add", "op": "add", "inputs": ["linear2", "input"], "kwargs": {}},
            {"id": "output", "op": "output_marker", "inputs": ["add"], "kwargs": {}},
        ]
        graph = GraphLink(nodes, output_id="output", trace=True)
        x = torch.randn(batch_size, input_dim)  # Input dimension is 15
        with self.assertRaises(ValueError) as context:
            out = graph(x)
        expected_error = f"Invalid weight shape ({input_dim}, 10) for node linear1, possible shapes: [({input_dim}, {input_dim})]"
        self.assertEqual(str(context.exception), expected_error, f"Expected error message '{expected_error}', got '{str(context.exception)}'")

if __name__ == '__main__':
    unittest.main()