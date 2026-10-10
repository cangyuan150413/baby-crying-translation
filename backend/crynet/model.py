"""Cry-reason classifier: train, save, load, predict."""
import wave
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import SAMPLE_RATE, extract_features
from .synthetic import make_dataset

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"


def pcm16_to_float(data: bytes):
    return np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float64) / 32768.0


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = pcm16_to_float(w.readframes(w.getnframes()))
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    if sr != SAMPLE_RATE:
        n = int(len(x) * SAMPLE_RATE / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return x


def load_wav_dataset(root):
    """Expect root/<label>/*.wav (16-bit PCM)."""
    X, y = [], []
    for d in sorted(Path(root).iterdir()):
        if d.is_dir():
            for f in d.glob("*.wav"):
                X.append(extract_features(read_wav(f)))
                y.append(d.name)
    return np.array(X), np.array(y)


def train(data_dir=None, seed=0):
    X, y = load_wav_dataset(data_dir) if data_dir else make_dataset(seed=seed)
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=seed))
    clf.fit(X, y)
    return clf


class CryClassifier:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    @classmethod
    def load(cls, path=MODEL_PATH):
        path = Path(path)
        if not path.exists():
            joblib.dump(train(), path)
        return cls(joblib.load(path))

    def predict(self, signal):
        p = self.pipeline.predict_proba([extract_features(signal)])[0]
        order = np.argsort(p)[::-1]
        classes = self.pipeline.classes_
        return {
            "label": str(classes[order[0]]),
            "confidence": float(p[order[0]]),
            "scores": {str(classes[i]): float(p[i]) for i in order},
        }
