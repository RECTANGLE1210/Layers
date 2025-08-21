import unittest
import torch

from units import ParamVec
from graph_link import Node, GraphLink
from ml_models import LinearRegressionBlock, LogisticRegressionBlock, SVMBlock


class UnsqueezeBlock(torch.nn.Module):
    def forward(self, x):
        return x.unsqueeze(-1)


class TestGraphStress(unittest.TestCase):
    def test_multiple_branches_concat(self):
        torch.manual_seed(42)
        batch, in_dim = 16, 7
        x = torch.randn(batch, in_dim)
        # Branches
        lin = LinearRegressionBlock(out_features=5)
        logit = LogisticRegressionBlock(out_features=3)
        svm = SVMBlock(out_features=4, mode="SVC")
        # ParamVecs
        pv_bias5 = ParamVec(shape=5, init="normal", name="b5")
        pv_bias3 = ParamVec(shape=3, init="normal", name="b3")
        pv_bias4 = ParamVec(shape=4, init="normal", name="b4")
        nodes = [
            Node(id="input", op="input", inputs=[], kwargs={}),
            Node(id="lin", op="custom_block", inputs=["input"], kwargs={"fn": lin}),
            Node(id="logit", op="custom_block", inputs=["input"], kwargs={"fn": logit}),
            Node(id="svm", op="custom_block", inputs=["input"], kwargs={"fn": svm}),
            Node(id="add1", op="add", inputs=["lin"], kwargs={"param": pv_bias5}),
            Node(id="add2", op="add", inputs=["logit"], kwargs={"param": pv_bias3}),
            Node(id="add3", op="add", inputs=["svm"], kwargs={"param": pv_bias4}),
            Node(id="concat", op="concat", inputs=["add1", "add2", "add3"], kwargs={}),
            Node(id="act", op="act", inputs=["concat"], kwargs={"act_type": "relu"}),
            Node(id="output", op="output_marker", inputs=["act"], kwargs={}),
        ]
        graph = GraphLink(nodes, output_id="output", trace=False)
        y = graph(x)
        self.assertEqual(y.shape, (batch, 5 + 3 + 4))
        self.assertTrue((y >= 0).all())  # relu

    def test_chain_operations(self):
        torch.manual_seed(123)
        batch, in_dim = 10, 8
        x = torch.randn(batch, in_dim)
        lin = LinearRegressionBlock(out_features=6)
        logit = LogisticRegressionBlock(out_features=4)
        svm = SVMBlock(out_features=2, mode="SVR")
        pv_a = ParamVec(shape=6, init="ones", name="a")
        pv_b = ParamVec(shape=6, init="uniform", name="b")
        nodes = [
            Node(id="input", op="input", inputs=[], kwargs={}),
            Node(id="lin", op="custom_block", inputs=["input"], kwargs={"fn": lin}),
            Node(id="add", op="add", inputs=["lin"], kwargs={"param": pv_a}),
            Node(id="mul", op="mul", inputs=["add"], kwargs={"param": pv_b}),
            Node(id="relu", op="act", inputs=["mul"], kwargs={"act_type": "relu"}),
            Node(id="logit", op="custom_block", inputs=["relu"], kwargs={"fn": logit}),
            Node(id="concat", op="concat", inputs=["relu", "logit"], kwargs={}),
            Node(id="svm", op="custom_block", inputs=["concat"], kwargs={"fn": svm}),
            Node(id="output", op="output_marker", inputs=["svm"], kwargs={}),
        ]
        graph = GraphLink(nodes, output_id="output", trace=False)
        y = graph(x)
        self.assertEqual(y.shape, (batch, 2))

    def test_random_paramvec_mismatch(self):
        torch.manual_seed(99)
        batch, in_dim = 5, 4
        x = torch.randn(batch, in_dim)
        lin = LinearRegressionBlock(out_features=3)
        logit = LogisticRegressionBlock(out_features=2)
        pv_good = ParamVec(shape=3, init="ones", name="good")
        pv_bad = ParamVec(shape=5, init="zeros", name="bad")  # wrong shape
        # Randomly insert the mismatch at add or mul
        import random
        mismatch_node = random.choice(["add", "mul"])
        nodes = [
            Node(id="input", op="input", inputs=[], kwargs={}),
            Node(id="lin", op="custom_block", inputs=["input"], kwargs={"fn": lin}),
            Node(id="add", op="add", inputs=["lin"], kwargs={"param": pv_bad if mismatch_node == "add" else pv_good}),
            Node(id="mul", op="mul", inputs=["add"], kwargs={"param": pv_bad if mismatch_node == "mul" else pv_good}),
            Node(id="relu", op="act", inputs=["mul"], kwargs={"act_type": "relu"}),
            Node(id="logit", op="custom_block", inputs=["relu"], kwargs={"fn": logit}),
            Node(id="output", op="output_marker", inputs=["logit"], kwargs={}),
        ]
        graph = GraphLink(nodes, output_id="output", trace=False)
        with self.assertRaises(RuntimeError) as cm:
            _ = graph(x)
        msg = str(cm.exception)
        self.assertIn(mismatch_node, msg)
        self.assertIn("shape", msg.lower())

    def test_nested_concat_add_mul_valid(self):
        torch.manual_seed(202)
        batch, in_dim = 6, 5
        x = torch.randn(batch, in_dim)
        # ParamVecs of different shape, all valid now
        pv1 = ParamVec(shape=5, init="ones", name="p1")
        pv2 = ParamVec(shape=5, init="ones", name="p2")  # changed shape from 3 to 5
        pv3 = ParamVec(shape=5, init="ones", name="p3")
        nodes = [
            Node(id="input", op="input", inputs=[], kwargs={}),
            Node(id="add1", op="add", inputs=["input"], kwargs={"param": pv3}),
            Node(id="mul1", op="mul", inputs=["add1"], kwargs={"param": pv3}),
            Node(id="add2", op="add", inputs=["input"], kwargs={"param": pv2}),
            Node(id="mul2", op="mul", inputs=["add2"], kwargs={"param": pv2}),
            Node(id="concat1", op="concat", inputs=["mul1", "mul2"], kwargs={}),
            Node(id="add3", op="add", inputs=["input"], kwargs={"param": pv1}),
            Node(id="mul3", op="mul", inputs=["add3"], kwargs={"param": pv1}),
            Node(id="concat2", op="concat", inputs=["concat1", "mul3"], kwargs={}),
            Node(id="output", op="output_marker", inputs=["concat2"], kwargs={}),
        ]
        graph = GraphLink(nodes, output_id="output", trace=False)
        y = graph(x)
        # Shape: (batch, 5+5+5 = 15)
        self.assertEqual(y.shape, (batch, 15))

    def test_nested_concat_add_mul_mismatch(self):
        torch.manual_seed(202)
        batch, in_dim = 6, 5
        x = torch.randn(batch, in_dim)
        # ParamVecs with mismatch shape for pv2
        pv1 = ParamVec(shape=2, init="ones", name="p1")
        pv2 = ParamVec(shape=3, init="ones", name="p2")  # original shape 3 to provoke mismatch
        pv3 = ParamVec(shape=5, init="ones", name="p3")
        nodes = [
            Node(id="input", op="input", inputs=[], kwargs={}),
            Node(id="add1", op="add", inputs=["input"], kwargs={"param": pv3}),
            Node(id="mul1", op="mul", inputs=["add1"], kwargs={"param": pv3}),
            Node(id="add2", op="add", inputs=["input"], kwargs={"param": pv2}),
            Node(id="mul2", op="mul", inputs=["add2"], kwargs={"param": pv2}),
            Node(id="concat1", op="concat", inputs=["mul1", "mul2"], kwargs={}),
            Node(id="add3", op="add", inputs=["input"], kwargs={"param": pv1}),
            Node(id="mul3", op="mul", inputs=["add3"], kwargs={"param": pv1}),
            Node(id="concat2", op="concat", inputs=["concat1", "mul3"], kwargs={}),
            Node(id="output", op="output_marker", inputs=["concat2"], kwargs={}),
        ]
        graph = GraphLink(nodes, output_id="output", trace=False)
        with self.assertRaises(RuntimeError) as cm:
            _ = graph(x)
        msg = str(cm.exception)
        self.assertIn("add2", msg)
        self.assertIn("shape", msg.lower())

    def test_large_batch(self):
        torch.manual_seed(77)
        batch, in_dim = 512, 12
        x = torch.randn(batch, in_dim)
        lin = LinearRegressionBlock(out_features=16)
        logit = LogisticRegressionBlock(out_features=8)
        svm = SVMBlock(out_features=4, mode="SVR")
        pv_bias16 = ParamVec(shape=16, init="normal", name="b16")
        pv_bias8 = ParamVec(shape=8, init="normal", name="b8")
        nodes = [
            Node(id="input", op="input", inputs=[], kwargs={}),
            Node(id="lin", op="custom_block", inputs=["input"], kwargs={"fn": lin}),
            Node(id="add1", op="add", inputs=["lin"], kwargs={"param": pv_bias16}),
            Node(id="logit", op="custom_block", inputs=["input"], kwargs={"fn": logit}),
            Node(id="add2", op="add", inputs=["logit"], kwargs={"param": pv_bias8}),
            Node(id="concat", op="concat", inputs=["add1", "add2"], kwargs={}),
            Node(id="svm", op="custom_block", inputs=["concat"], kwargs={"fn": svm}),
            Node(id="output", op="output_marker", inputs=["svm"], kwargs={}),
        ]
        graph = GraphLink(nodes, output_id="output", trace=False)
        y = graph(x)
        self.assertEqual(y.shape, (batch, 4))
import unittest
import torch

from units import ParamVec
from graph_link import Node, GraphLink
from ml_models import LinearRegressionBlock, LogisticRegressionBlock, SVMBlock


class TestComplexGraphlink(unittest.TestCase):
    def test_end_to_end_with_mismatch_branch(self):
        torch.manual_seed(0)

        # ====== Input ======
        batch, in_dim = 12, 6
        x = torch.randn(batch, in_dim)

        # ====== ML Blocks (lazy) ======
        # Linear will infer in_features=6, project to 8
        lin = LinearRegressionBlock(out_features=8)
        # Logistic will infer in_features=6, project to 3 then sigmoid
        logit = LogisticRegressionBlock(out_features=3)
        # SVM (SVR mode) will infer from concat(8 + 3 = 11) → out_features=2
        svm = SVMBlock(out_features=2, mode="SVR")

        # ====== ParamVecs ======
        pv_bias8 = ParamVec(shape=8, init="normal", init_kwargs={"mean": 0.0, "std": 0.1}, name="bias8")
        pv_scale8 = ParamVec(shape=8, init="ones", name="scale8")
        pv_dot8   = ParamVec(shape=8, init="uniform", init_kwargs={"a": -0.5, "b": 0.5}, name="dot8")

        # A wrong-sized bias (7) to provoke mismatch in a separate tiny graph below
        pv_wrong7 = ParamVec(shape=7, init="zeros", name="wrong7")

        # ====== Graph (main) ======
        # Flow:
        # input --(custom Linear)-> l1 --(+ bias8)-> l2 --(* scale8)-> l3 --(ReLU)-> l4
        # input --(custom Logistic)-> p
        # concat(l4, p) -> c  # shape: (batch, 8 + 3 = 11)
        # custom SVM(SVR) on c -> s (batch, 2)
        # softmax over last dim -> sm
        # dot(l1, dot8) -> d (batch,)  # an auxiliary branch
        # concat(sm, d.unsqueeze(1)) -> y (batch, 3)
        # threshold(y, 0.0) -> y_bin (float 0/1, batch, 3) -> output

        nodes = [
            Node(id="input", op="input", inputs=[], kwargs={}),

            # branch A
            Node(id="lin", op="custom_block", inputs=["input"], kwargs={"fn": lin}),
            Node(id="add_bias", op="add", inputs=["lin"], kwargs={"param": pv_bias8}),
            Node(id="mul_scale", op="mul", inputs=["add_bias"], kwargs={"param": pv_scale8}),
            Node(id="relu", op="act", inputs=["mul_scale"], kwargs={"act_type": "relu"}),

            # branch B
            Node(id="logistic", op="custom_block", inputs=["input"], kwargs={"fn": logit}),  # (batch, 3)

            # fuse
            Node(id="concat", op="concat", inputs=["relu", "logistic"], kwargs={}),   # (batch, 11)

            # head
            Node(id="svr", op="custom_block", inputs=["concat"], kwargs={"fn": svm}),       # (batch, 2)
            Node(id="softmax", op="act", inputs=["svr"], kwargs={"act_type": "softmax"}),

            # aux dot branch from lin (before bias/scale) to scalar and concat to 3 channels
            Node(id="dot_score", op="dot", inputs=["lin"], kwargs={"param": pv_dot8}),  # (batch,)
            # bring (batch,) to (batch,1) by simple trick: concat with itself after a sign -> but simpler:
            # use act: tanh on a reshape is not supported here; instead we can concat via custom tiny module,
            # but keep it simple—PyTorch can unsqueeze inside a custom lambda fn:
            Node(id="unsq", op="custom_block", inputs=["dot_score"], kwargs={"fn": UnsqueezeBlock()}),

            Node(id="concat3", op="concat", inputs=["softmax", "unsq"], kwargs={}),    # (batch, 3)
            Node(id="thresh", op="threshold", inputs=["concat3"], kwargs={"threshold": 0.0}),

            Node(id="output", op="output_marker", inputs=["thresh"], kwargs={}),
        ]

        graph = GraphLink(nodes, output_id="output", trace=True)
        y = graph(x)

        # ====== Assertions: shapes ======
        self.assertEqual(y.shape, (batch, 3))                 # after threshold
        self.assertTrue(((y == 0) | (y == 1)).all())          # 0/1 values (float)
        # Blocks should have lazily inferred in_features
        self.assertEqual(lin.in_features, in_dim)
        self.assertEqual(logit.in_features, in_dim)
        # Concat fed SVM with 11-dim features
        self.assertEqual(svm.in_features, 11)

        # Full shapes recorded on blocks
        self.assertEqual(lin.input_shape, (batch, in_dim))
        self.assertEqual(lin.output_shape, (batch, 8))
        self.assertEqual(logit.output_shape, (batch, 3))
        # Feature-only views
        self.assertEqual(lin.input_shape_feat, (in_dim,))
        self.assertEqual(lin.output_shape_feat, (8,))
        self.assertEqual(logit.output_shape_feat, (3,))

        # ====== Mismatch branch: build a tiny graph that must fail (bias of size 7 on a 8-D tensor) ======
        bad_nodes = [
            Node(id="input", op="input", inputs=[], kwargs={}),
            Node(id="lin", op="custom_block", inputs=["input"], kwargs={"fn": LinearRegressionBlock(out_features=8)}),
            Node(id="bad_add", op="add", inputs=["lin"], kwargs={"param": pv_wrong7}),   # wrong size!
            Node(id="out", op="output_marker", inputs=["bad_add"], kwargs={}),
        ]
        bad_graph = GraphLink(bad_nodes, output_id="out", trace=False)

        with self.assertRaises(RuntimeError) as cm:
            _ = bad_graph(torch.randn(4, in_dim))
        msg = str(cm.exception)
        # Message should carry node id and mention shapes
        self.assertIn("bad_add", msg)
        self.assertTrue("mismatch" in msg.lower() or "Invalid" in msg or "shape" in msg)

if __name__ == "__main__":
    unittest.main()