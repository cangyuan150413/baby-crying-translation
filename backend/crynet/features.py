"""Numpy-only acoustic feature extraction for infant cry audio."""
import numpy as np

SAMPLE_RATE = 16000
FRAME = 1024
HOP = 512
BANDS = [(0, 300), (300, 600), (600, 1200), (1200, 2400), (2400, 4800), (4800, 8000)]


def _frames(x):
    if len(x) < FRAME:
        x = np.pad(x, (0, FRAME - len(x)))
    n = 1 + (len(x) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx] * np.hanning(FRAME)


def _pitch(frame, sr):
    ac = np.correlate(frame, frame, "full")[FRAME - 1:]
    lo, hi = int(sr / 1000), int(sr / 200)
    seg = ac[lo:hi]
    if ac[0] <= 1e-9 or len(seg) == 0:
        return 0.0, 0.0
    k = int(np.argmax(seg))
    return sr / (lo + k), float(seg[k] / ac[0])


def extract_features(signal, sr=SAMPLE_RATE):
    """Return a fixed-length feature vector for a mono signal."""
    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean() if len(x) else x
    peak = np.max(np.abs(x)) if len(x) else 0.0
    if peak > 0:
        x = x / peak
    fr = _frames(x)
    spec = np.abs(np.fft.rfft(fr, axis=1)) ** 2 + 1e-12
    freqs = np.fft.rfftfreq(FRAME, 1.0 / sr)
    total = spec.sum(1)
    centroid = (spec * freqs).sum(1) / total
    bw = np.sqrt((spec * (freqs - centroid[:, None]) ** 2).sum(1) / total)
    rolloff = freqs[np.argmax(np.cumsum(spec, 1) >= 0.85 * total[:, None], axis=1)]
    rms = np.sqrt((fr ** 2).mean(1))
    zcr = (np.diff(np.sign(fr), axis=1) != 0).mean(1)
    pitches = np.array([_pitch(f, sr) for f in fr])
    f0, voiced = pitches[:, 0], pitches[:, 1]
    mask = voiced > 0.3
    f0v = f0[mask] if mask.any() else np.zeros(1)
    bands = np.array([spec[:, (freqs >= a) & (freqs < b)].sum(1) for a, b in BANDS]).T
    bands = bands / bands.sum(1, keepdims=True)
    active = rms > 0.1 * (rms.max() + 1e-12)
    feats = [
        centroid.mean(), centroid.std(), bw.mean(), rolloff.mean(),
        rms.mean(), rms.std(), zcr.mean(),
        f0v.mean(), f0v.std(), f0v.max() - f0v.min(), mask.mean(),
        active.mean(), len(x) / sr,
    ]
    feats.extend(bands.mean(0))
    return np.array(feats, dtype=np.float64)
