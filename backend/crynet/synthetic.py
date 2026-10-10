"""Synthetic cry generator used when no labelled recordings are available.

Profiles are rough, illustrative approximations of cry characteristics
(pitch, rhythm, harshness); replace with real labelled data for production.
"""
import numpy as np
from .features import SAMPLE_RATE

LABELS = ["hungry", "tired", "discomfort", "pain", "burp"]
# f0 (Hz), vibrato depth, burst rate (bursts/s), noise level
PROFILES = {
    "hungry": (450, 0.05, 1.0, 0.05),
    "tired": (380, 0.03, 0.5, 0.10),
    "discomfort": (500, 0.10, 1.5, 0.20),
    "pain": (800, 0.20, 3.0, 0.15),
    "burp": (300, 0.02, 2.0, 0.35),
}


def make_cry(label, rng, seconds=3.0, sr=SAMPLE_RATE):
    f0, vib, rate, noise = PROFILES[label]
    f0 *= rng.uniform(0.9, 1.1)
    t = np.arange(int(seconds * sr)) / sr
    f = f0 * (1 + vib * np.sin(2 * np.pi * rng.uniform(4, 6) * t))
    phase = 2 * np.pi * np.cumsum(f) / sr
    sig = sum(np.sin(h * phase) / h for h in range(1, 8))
    env = np.clip(np.sin(np.pi * rate * t), 0, None) ** 0.5
    sig = sig * env + noise * rng.standard_normal(len(t))
    return sig / (np.max(np.abs(sig)) + 1e-9)


def make_dataset(n_per_class=60, seed=0):
    from .features import extract_features
    rng = np.random.default_rng(seed)
    X, y = [], []
    for lab in LABELS:
        for _ in range(n_per_class):
            X.append(extract_features(make_cry(lab, rng)))
            y.append(lab)
    return np.array(X), np.array(y)
