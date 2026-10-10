"""Realtime server. WebSocket clients:
  /ws/analyze   - send binary 16 kHz mono PCM16 chunks; receive JSON results
  /ws/subscribe - mobile app: receive JSON results broadcast from any analyzer
POST /analyze   - raw PCM16 body -> JSON result (also broadcast to subscribers)
"""
import time

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect

from crynet.features import SAMPLE_RATE
from crynet.model import CryClassifier, pcm16_to_float

app = FastAPI(title="Baby cry analyzer")
clf = CryClassifier.load()
subscribers = set()
WINDOW = 2 * SAMPLE_RATE
MIN_RMS = 0.01


def analyze(pcm: bytes):
    x = pcm16_to_float(pcm)
    if len(x) == 0 or (x ** 2).mean() ** 0.5 < MIN_RMS:
        return {"type": "silence", "ts": time.time()}
    return {"type": "cry", "ts": time.time(), **clf.predict(x)}


async def broadcast(msg):
    for ws in list(subscribers):
        try:
            await ws.send_json(msg)
        except Exception:
            subscribers.discard(ws)


@app.post("/analyze")
async def analyze_http(request: Request):
    msg = analyze(await request.body())
    await broadcast(msg)
    return msg


@app.websocket("/ws/analyze")
async def ws_analyze(ws: WebSocket):
    await ws.accept()
    buf = bytearray()
    try:
        while True:
            buf += await ws.receive_bytes()
            while len(buf) >= WINDOW * 2:
                msg = analyze(bytes(buf[: WINDOW * 2]))
                del buf[: WINDOW * 2]
                await ws.send_json(msg)
                await broadcast(msg)
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/subscribe")
async def ws_subscribe(ws: WebSocket):
    await ws.accept()
    subscribers.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        subscribers.discard(ws)
