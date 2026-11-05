"""Minimal FastAPI service: upload a photo, get back the rectified scan or just the detected corners.

Run: uvicorn docscan.api:app --port 8000
  POST /scan    multipart 'file' -> image/png scan; header X-Scan-Info holds corners, method, score, rotation, skew
  POST /detect  multipart 'file' -> JSON with corners, method, score and the ranked candidates
"""
import json

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from .detect import detect_document
from .pipeline import scan

app = FastAPI(title="docscan", version="0.1.0")


async def _read(file: UploadFile) -> np.ndarray:
    data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="could not decode image")
    return img


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    det = detect_document(await _read(file))
    return JSONResponse({
        "corners": det.quad.round(1).tolist(), "method": det.method, "score": round(det.score, 4),
        "candidates": [{"method": c["method"], "score": round(c["score"], 4)} for c in det.all_quads[:5]],
    })


@app.post("/scan")
async def scan_endpoint(file: UploadFile = File(...), orient: bool = True, deskew: bool = True):
    res = scan(await _read(file), orient=orient, fix_skew=deskew)
    ok, buf = cv2.imencode(".png", res.image)
    info = {"corners": res.detection.quad.round(1).tolist(), "method": res.detection.method,
            "score": round(res.detection.score, 4), "rotation_quarter_turns": res.rotation_k,
            "skew_deg": round(res.skew_deg, 2), "size": list(res.size)}
    return Response(content=buf.tobytes(), media_type="image/png", headers={"X-Scan-Info": json.dumps(info)})
