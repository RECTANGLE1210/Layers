import torch
import unittest
from graph_link import GraphLink
from units import ParamVec

class TestGraphLink(unittest.TestCase):
    def test_dag(self):
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

    def test_residual_block(self):
        input_dim = 8
        batch_size = 32  # Example batch size

        # ParamVec for both linears
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

if __name__ == "__main__":
    print("Running examples...")

    # GraphLink example: DAG
    x = torch.randn(5,2)
    weight1 = ParamVec(shape=None, init="xavier_uniform", init_kwargs={"fan_out": 4})
    bias1 = ParamVec(shape=None, init="zeros")
    weight2 = ParamVec(shape=None, init="xavier_uniform", init_kwargs={"fan_out": 4})
    bias2 = ParamVec(shape=None, init="zeros")
    nodes = [
        {"id": "in", "op": "input_marker", "inputs": [], "kwargs": {}},
        {"id": "linear1", "op": "linear", "inputs": ["in"], "kwargs": {"dim_out": 4, "weight": weight1, "bias": bias1}},
        {"id": "linear2", "op": "linear", "inputs": ["in"], "kwargs": {"dim_out": 4, "weight": weight2, "bias": bias2}},
        {"id": "concat", "op": "concat", "inputs": ["linear1", "linear2"], "kwargs": {}},
        {"id": "softmax", "op": "act", "inputs": ["concat"], "kwargs": {"act_type": "softmax"}},
        {"id": "out", "op": "output_marker", "inputs": ["softmax"], "kwargs": {}},
    ]
    graph = GraphLink(nodes, output_id="out", trace=True)
    out = graph(x)
    print(f"GraphLink output shape: {out.shape}")

    # Random GraphLink stress demo
    import random

    def build_random_graph(
        in_dim: int = 8,
        num_nodes: int = 30,
        seed: int = 123,
    ):
        random.seed(seed)
        torch.manual_seed(seed)
        nodes = [
            {"id": "in", "op": "input_marker", "inputs": [], "kwargs": {}}
        ]
        dims = {"in": in_dim}

        def pick_prev_ids(k: int = 1):
            ids = list(dims.keys())
            chosen = random.sample(ids, k=min(k, len(ids)))
            return chosen

        for i in range(num_nodes):
            nid = f"n{i}"
            op = random.choices(
                population=["linear", "add", "mul", "act", "concat"],
                weights=[0.35, 0.2, 0.2, 0.15, 0.10],
                k=1,
            )[0]
            if op == "linear":
                src = pick_prev_ids(1)[0]
                dim_out = random.randint(2, 16)
                w = ParamVec(shape=None, init="xavier_uniform", init_kwargs={"fan_out": dim_out})
                b = ParamVec(shape=None, init="zeros")
                nodes.append({"id": nid, "op": "linear", "inputs": [src], "kwargs": {"dim_out": dim_out, "weight": w, "bias": b}})
                dims[nid] = dim_out
            elif op in ("add", "mul"):
                src = pick_prev_ids(1)[0]
                p = ParamVec(shape=None, init="normal", init_kwargs={"mean": 0.0, "std": 0.02})
                nodes.append({"id": nid, "op": op, "inputs": [src], "kwargs": {"param": p}})
                dims[nid] = dims[src]
            elif op == "act":
                src = pick_prev_ids(1)[0]
                act_type = random.choice(["relu", "tanh", "softmax"])
                nodes.append({"id": nid, "op": "act", "inputs": [src], "kwargs": {"act_type": act_type}})
                dims[nid] = dims[src]
            elif op == "concat" and len(dims) >= 2:
                k = random.randint(2, min(3, len(dims)))
                srcs = pick_prev_ids(k)
                total = sum(dims[s] for s in srcs)
                nodes.append({"id": nid, "op": "concat", "inputs": srcs, "kwargs": {}})
                dims[nid] = total
            else:
                src = pick_prev_ids(1)[0]
                dim_out = random.randint(2, 16)
                w = ParamVec(shape=None, init="xavier_uniform", init_kwargs={"fan_out": dim_out})
                b = ParamVec(shape=None, init="zeros")
                nodes.append({"id": nid, "op": "linear", "inputs": [src], "kwargs": {"dim_out": dim_out, "weight": w, "bias": b}})
                dims[nid] = dim_out

        out_src = random.choice(list(dims.keys()))
        nodes.append({"id": "out", "op": "output_marker", "inputs": [out_src], "kwargs": {}})
        return nodes, "out", dims[out_src]

    # Run a random DAG stress test demo
    nodes, out_id, out_dim = build_random_graph(in_dim=12, num_nodes=50, seed=2025)
    graph_rand = GraphLink(nodes, output_id=out_id, trace=True)
    x = torch.randn(12)
    out = graph_rand(x)
    print(f"[Random DAG] num_nodes={len(nodes)} -> output shape: {out.shape} (expected ({out_dim},))")

    # Run tests
    unittest.main(exit=False)