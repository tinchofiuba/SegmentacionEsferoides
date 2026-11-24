import hashlib
import json

from u_resnet_segesferoides.make_index import build_index


def test_index_lists_runs_with_sha256(tmp_path):
    run = tmp_path / "run_1"
    run.mkdir()
    (run / "best_model.onnx").write_bytes(b"abc")
    (run / "best_model.yaml").write_text("schema: 1\n")
    path = build_index([str(run)], str(tmp_path / "out"), notes="mejor")
    entry = json.load(open(path))["models"][0]
    assert entry["name"] == "run_1" and entry["onnx"] == "run_1.onnx"
    assert entry["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert (tmp_path / "out" / "run_1.yaml").exists()


def test_name_overrides_run_name(tmp_path):
    run = tmp_path / "run_1"
    run.mkdir()
    (run / "best_model.onnx").write_bytes(b"abc")
    (run / "best_model.yaml").write_text("schema: 1\n")
    path = build_index([str(run)], str(tmp_path / "out"), name="Modelo_X")
    entry = json.load(open(path))["models"][0]
    assert entry["name"] == "Modelo_X" and entry["onnx"] == "Modelo_X.onnx"
    assert (tmp_path / "out" / "Modelo_X.yaml").exists()
