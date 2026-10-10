import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient  # noqa: E402

from crynet.synthetic import LABELS, make_cry  # noqa: E402
from crynet.model import CryClassifier  # noqa: E402
import server  # noqa: E402


def pcm(x):
    return (x * 32767).astype("<i2").tobytes()


def test_classifier_accuracy():
    rng = np.random.default_rng(123)
    ok = n = 0
    for lab in LABELS:
        for _ in range(10):
            ok += server.clf.predict(make_cry(lab, rng))["label"] == lab
            n += 1
    assert ok / n > 0.8


def test_http_and_ws():
    c = TestClient(server.app)
    audio = pcm(make_cry("pain", np.random.default_rng(1)))
    with c.websocket_connect("/ws/subscribe") as sub:
        r = c.post("/analyze", content=audio).json()
        assert r["type"] == "cry" and r["label"] in LABELS
        assert sub.receive_json()["label"] == r["label"]
    assert c.post("/analyze", content=b"\0" * 64000).json()["type"] == "silence"
