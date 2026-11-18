import onnx
import segmentation_models_pytorch as smp
import torch

from u_resnet_segesferoides.export_onnx import export_run_to_onnx


def test_export_has_dynamic_hw_and_valid_graph(tmp_path):
    model = smp.Unet("resnet34", encoder_weights=None, classes=4)
    torch.save(model.state_dict(), tmp_path / "best_model.pth")
    config = {"training": {"num_classes": 4}}
    out = export_run_to_onnx(str(tmp_path), config)
    graph = onnx.load(out)
    onnx.checker.check_model(graph)
    dims = [d.dim_param or d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]
    assert dims == ["batch", 3, "height", "width"]
