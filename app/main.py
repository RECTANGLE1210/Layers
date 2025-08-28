import os, json
import uuid
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

from components.graph_link import Node, GraphLink
from components.ml_models import LinearRegressionBlock, LogisticRegressionBlock, SVMBlock
from components.units import ParamVec

# ---------------- FastAPI init ----------------
app = FastAPI(title="GraphLink API", version="1.0")

# ---------------- Registry ----------------
MODEL_REGISTRY: Dict[str, GraphLink] = {}

# ---------------- Storage Directory ----------------
STORAGE_DIR = "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

# ---------------- Schemas ----------------
class ParamVecSchema(BaseModel):
    shape: Any = None
    init: str = "xavier_uniform"
    init_kwargs: Dict[str, Any] = {}
    name: str = None
    freeze: bool = False

class NodeSchema(BaseModel):
    id: str
    op: str
    inputs: List[str]
    kwargs: Dict[str, Any]

class GraphSchema(BaseModel):
    nodes: List[NodeSchema]
    output_id: str

class ForwardRequest(BaseModel):
    input: List[List[float]]

# ---------------- Helpers ----------------
def build_graphlink_from_json(graph_json: GraphSchema) -> GraphLink:
    nodes = []
    for n in graph_json.nodes:
        kwargs = {}
        for k, v in n.kwargs.items():
            if isinstance(v, dict) and "init" in v:  # ParamVec
                kwargs[k] = ParamVec(
                    shape=v.get("shape", None),
                    init=v.get("init", "xavier_uniform"),
                    init_kwargs=v.get("init_kwargs", {}),
                    name=v.get("name", None),
                    freeze=v.get("freeze", False),
                )
            else:
                kwargs[k] = v

        # Nếu là custom_block thì khởi tạo ML block từ "fn"
        if n.op == "custom_block":
            fn_name = kwargs.pop("fn", None)
            if fn_name == "LinearRegressionBlock":
                block_instance = LinearRegressionBlock(**kwargs)
            elif fn_name == "LogisticRegressionBlock":
                block_instance = LogisticRegressionBlock(**kwargs)
            elif fn_name == "SVMBlock":
                block_instance = SVMBlock(**kwargs)
            else:
                raise ValueError(f"Unsupported custom block type: {fn_name}")
            kwargs = {"fn": block_instance}
        nodes.append(Node(id=n.id, op=n.op, inputs=n.inputs, kwargs=kwargs))
    return GraphLink(nodes, output_id=graph_json.output_id)

# Save model metadata as JSON in storage/{model_id}.json and return filepath
def save_model_metadata(model_id: str, graph: GraphSchema, model: GraphLink) -> str:
    num_params = sum(p.numel() for p in model.parameters() if hasattr(p, "numel"))
    nodes_info = [
        {"id": n.id, "op": n.op, "inputs": n.inputs}
        for n in model.nodes.values()
    ]
    metadata = {
        "model_id": model_id,
        "num_params": num_params,
        "nodes": nodes_info,
        "output_id": model.output_id,
        "topo_order": model.topo_order,
        "raw_config": graph.dict(),
    }
    filepath = os.path.join(STORAGE_DIR, f"{model_id}.json")
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)
    return filepath

# ---------------- API endpoints ----------------
@app.post("/models")
def create_model(graph: GraphSchema):
    try:
        model = build_graphlink_from_json(graph)
        model_id = str(uuid.uuid4())
        MODEL_REGISTRY[model_id] = model
        filepath = save_model_metadata(model_id, graph, model)
        input_shape = None
        for n in graph.nodes:
            if n.op == "input":
                input_shape = n.kwargs.get("input_shape", None)
                break
        return {
            "status": "succeeded",
            "model_id": model_id,
            "metadata_path": filepath,
            "input_shape": input_shape,
        }
    except Exception as e:
        return {"status": "failed", "error": str(e)}

@app.get("/models/{model_id}")
def get_model_info(model_id: str):
    filepath = os.path.join(STORAGE_DIR, f"{model_id}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                metadata = json.load(f)
            metadata["status"] = "succeeded"
            metadata["metadata_path"] = filepath
            input_shape = next((n["kwargs"].get("input_shape") for n in metadata["raw_config"]["nodes"] if n["op"] == "input"),None)
            dummy_tensor = torch.rand(input_shape)
            # 2. Lấy raw_config ra và parse thành GraphSchema
            graph_json = GraphSchema(**metadata["raw_config"])
            # 3. Build model từ JSON
            model = build_graphlink_from_json(graph_json)
            _ = model(dummy_tensor)
            total_params = sum(p.numel() for p in model.parameters())
            metadata["num_params"] = total_params
            with open(filepath, "w") as f:
                json.dump(metadata, f, indent=2)
            return metadata
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    else:
        return {"status": "failed", "error": "Model not found"}

@app.post("/models/{model_id}/forward")
def forward_model(model_id: str, request: ForwardRequest):
    model = MODEL_REGISTRY.get(model_id)
    if model is None:
        return {"status": "failed", "error": "Model not found"}
    try:
        x = torch.tensor(request.input, dtype=torch.float32)
        y = model(x)
        return {"status": "succeeded", "output": y.tolist()}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

@app.delete("/models/{model_id}")
def delete_model(model_id: str):
    filepath = os.path.join(STORAGE_DIR, f"{model_id}.json")
    deleted_file = None
    if model_id in MODEL_REGISTRY:
        del MODEL_REGISTRY[model_id]
        if os.path.exists(filepath):
            os.remove(filepath)
            deleted_file = filepath
        return {"status": "succeeded", "message": f"Model {model_id} deleted", "deleted_file": deleted_file}
    else:
        return {"status": "failed", "error": "Model not found"}