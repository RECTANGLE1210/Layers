import torch
from torchinfo import summary

from units import ParamVec
from graph_link import Node, GraphLink
from ml_models import LinearRegressionBlock, LogisticRegressionBlock, SVMBlock


def build_multibranch_concat_graph():
    lin = LinearRegressionBlock(out_features=5)   # lazy init
    logit = LogisticRegressionBlock(out_features=3)  # lazy init
    svm = SVMBlock(out_features=4, mode="SVC")   # lazy init

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
    return GraphLink(nodes, output_id="output", trace=False)


if __name__ == "__main__":
    model = build_multibranch_concat_graph()

    # ép lazy init bằng cách chạy 1 forward pass
    x = torch.randn(2, 7)   # batch=2, in_features=7
    _ = model(x)

    # giờ mới gọi summary, tham số đã được tạo
    summary(model, input_size=(2, 7), col_names=("input_size", "output_size", "num_params"))