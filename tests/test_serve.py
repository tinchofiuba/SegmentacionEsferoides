import io
import json
import os
import time

from u_resnet_segesferoides.serve import class_definitions, list_runs, serve_loop


class StubBackend:
    def info(self):
        return {"protocol": 1, "name": "stub"}

    def list_models(self):
        return []

    def load(self, model):
        if model == "malo":
            raise FileNotFoundError("no existe")
        return {"description": model}

    def predict(self, image, magnification, bbox, out):
        return {"mask": out}


def _run(requests):
    fin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    fout = io.StringIO()
    serve_loop(StubBackend(), fin, fout)
    return [json.loads(line) for line in fout.getvalue().splitlines()]


def test_serve_loop_roundtrip_and_errors_do_not_kill_server():
    resp = _run([
        {"id": 1, "cmd": "info"},
        {"id": 2, "cmd": "load", "model": "malo"},
        {"id": 3, "cmd": "nope"},
        {"id": 4, "cmd": "predict", "image": "a.png", "magnification": "10x", "bbox": None, "out": "o.png"},
        {"id": 5, "cmd": "shutdown"},
        {"id": 6, "cmd": "info"},  # posterior al shutdown: no se atiende
    ])
    assert [r["id"] for r in resp] == [1, 2, 3, 4, 5]
    assert resp[0]["ok"] and resp[0]["result"]["protocol"] == 1
    assert not resp[1]["ok"] and "no existe" in resp[1]["error"]
    assert not resp[2]["ok"] and "desconocido" in resp[2]["error"]
    assert resp[3]["result"] == {"mask": "o.png"}
    assert resp[4]["ok"]


def test_serve_loop_survives_garbage_line():
    fin = io.StringIO("esto no es json\n" + json.dumps({"id": 2, "cmd": "info"}) + "\n")
    fout = io.StringIO()
    serve_loop(StubBackend(), fin, fout)
    lines = [json.loads(line) for line in fout.getvalue().splitlines()]
    assert not lines[0]["ok"] and lines[1]["ok"]


def test_class_definitions_match_dataset_mapping():
    classes = {c["id"]: c["color"] for c in class_definitions()}
    # MASK_COLOR_TO_CLASS está en BGR: verde=1, amarillo=2, cyan=3 → RGB
    assert classes == {1: [0, 255, 0], 2: [255, 255, 0], 3: [0, 255, 255]}


def test_list_runs_filters_and_sorts_newest_first(tmp_path):
    now = time.time()

    def make(name, finished, mtime):
        d = tmp_path / name
        d.mkdir()
        (d / "config.yaml").write_text("x: 1\n")
        (d / "best_model.pth").write_bytes(b"")
        if finished:
            (d / "metrics.txt").write_text("ok")
        os.utime(d, (mtime, mtime))

    make("viejo", True, now - 1000)
    make("en_curso", False, now)
    (tmp_path / "sin_modelo").mkdir()
    (tmp_path / "sin_modelo" / "config.yaml").write_text("x: 1\n")

    runs = list_runs(str(tmp_path))
    assert [r["name"] for r in runs] == ["en_curso", "viejo"]
    assert [r["finished"] for r in runs] == [False, True]
    assert list_runs(str(tmp_path / "no_existe")) == []
