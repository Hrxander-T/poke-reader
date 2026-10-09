"""
app.py - web API for the PokeMMO card reader.

    POST /read     multipart form field "file" = PNG/JPG (cropped card OR full screenshot) -> JSON
    GET  /health   quick "is it up" check

Settings (all optional, via environment variables):
    DATA_PATH        pokemon_data.json         TEMPLATE_PATH   card_template.npz
    ALLOWED_ORIGINS  comma list of website addresses allowed to call this from a browser (default: *)
    API_KEY          if set, every /read call must send header  X-API-Key: <value>
    MAX_MB           biggest upload (default 10)     MAX_PIXELS  biggest image (default 12,000,000)
    MAX_CONCURRENT   how many images are read at the same time (default 2)
"""
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

import pokemmo_reader as pr

DATA_PATH = os.getenv('DATA_PATH', 'pokemon_data.json')
TEMPLATE_PATH = os.getenv('TEMPLATE_PATH', 'card_template.npz')
API_KEY = os.getenv('API_KEY', '')
MAX_BYTES = int(float(os.getenv('MAX_MB', '10')) * 1024 * 1024)
MAX_PIXELS = int(os.getenv('MAX_PIXELS', '12000000'))
ORIGINS = [o.strip() for o in os.getenv('ALLOWED_ORIGINS', '*').split(',') if o.strip()]

_slots = threading.BoundedSemaphore(int(os.getenv('MAX_CONCURRENT', '2')))


@asynccontextmanager
async def lifespan(app):
    pr.load(DATA_PATH, TEMPLATE_PATH)      # once, at startup
    yield


app = FastAPI(title='PokeMMO card reader', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=['GET', 'POST'], allow_headers=['*'])


@app.get('/health')
def health():
    return {'ok': True, 'pokemon_loaded': len(pr.pe.DATA)}


@app.post('/read')
def read(file: UploadFile = File(...), x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, 'missing or wrong X-API-Key')
    raw = file.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f'file too big (max {MAX_BYTES // 1024 // 1024} MB)')
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, 'not a readable image (use PNG or JPG)')
    if img.shape[0] * img.shape[1] > MAX_PIXELS:
        raise HTTPException(413, 'image too large')
    try:
        with _slots:                       # OCR is CPU-heavy: limit how many run at once
            card = pr.read_image(img, file.filename or '')
    except Exception:
        raise HTTPException(500, 'could not process this image')
    return Response(content=pr.to_json(card), media_type='application/json')
