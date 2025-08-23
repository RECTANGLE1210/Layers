import torch
from torchinfo import summary

from units import ParamVec
from graph_link import Node, GraphLink
from ml_models import LinearRegressionBlock, LogisticRegressionBlock, SVMBlock

def build_paramvec_combo_graph():
    lin = LinearRegressionBlock(out_features=5)
    logit = LogisticRegressionBlock(out_features=3)
    svm = SVMBlock(out_features=4, mode="SVR")

    # ParamVec để bias/add
    pv_bias5 = ParamVec(shape=5, init="normal", name="bias5")
    pv_bias3 = ParamVec(shape=3, init="normal", name="bias3")
    pv_bias4 = ParamVec(shape=4, init="normal", name="bias4")

    # ParamVec để nhân/ghép
    pv_mul5 = ParamVec(shape=5, init="ones", name="scale5")
    pv_mul3 = ParamVec(shape=3, init="ones", name="scale3")

    pv_fusion = ParamVec(shape=5, init="normal", name="fusion_bias")

    nodes = [
        Node(id="input", op="input", inputs=[], kwargs={}),
        # ML blocks
        Node(id="lin", op="custom_block", inputs=["input"], kwargs={"fn": lin}),
        Node(id="logit", op="custom_block", inputs=["input"], kwargs={"fn": logit}),
        Node(id="svm", op="custom_block", inputs=["input"], kwargs={"fn": svm}),
        # Add bias
        Node(id="lin_b", op="add", inputs=["lin"], kwargs={"param": pv_bias5}),
        Node(id="logit_b", op="add", inputs=["logit"], kwargs={"param": pv_bias3}),
        Node(id="svm_b", op="add", inputs=["svm"], kwargs={"param": pv_bias4}),
        # Multiply scale
        Node(id="lin_s", op="mul", inputs=["lin_b"], kwargs={"param": pv_mul5}),
        Node(id="logit_s", op="mul", inputs=["logit_b"], kwargs={"param": pv_mul3}),
        # Add fusion bias
        Node(id="fusion_add", op="add", inputs=["lin_s"], kwargs={"param": pv_fusion}),
        # Concat tất cả lại
        Node(id="concat", op="concat", inputs=["lin_s", "logit_s", "svm_b", "fusion_add"], kwargs={}),
        Node(id="act", op="act", inputs=["concat"], kwargs={"act_type": "relu"}),
        Node(id="output", op="output_marker", inputs=["act"], kwargs={}),
    ]
    return GraphLink(nodes, output_id="output", trace=False)


if __name__ == "__main__":
    model = build_paramvec_combo_graph()

    # ép lazy init
    x = torch.randn(2, 7)   # batch=2, in_features=7
    _ = model(x)

    summary(model, input_size=(2, 7),
            col_names=("input_size", "output_size", "num_params"))