"""Face recognition encoder API.

Stateless face detector/encoder service intended to be called by the
Home Assistant "intersvyaz" integration as a remote alternative to its local
dlib backend (which crashes with SIGILL on CPUs without AVX support).

This service intentionally does NOT own a face library (no named persons,
no on-disk photo storage, no matching against known people). It only detects
faces in an uploaded image and returns 128-d dlib ResNet descriptors; storing
known faces and comparing distances/thresholds stays entirely on the caller
side, so there is a single source of truth for who is "known".
"""
import io
import logging
import os
from typing import Optional

import face_recognition
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face_recognize_service")

ENGINE_ID = "remote_dlib_resnet_v1"
ENCODING_SIZE = 128
DEFAULT_ENROLL_JITTERS = 2
DEFAULT_RECOGNIZE_JITTERS = 1
MAX_JITTERS = 10

API_TOKEN = os.environ.get("FACE_API_TOKEN", "").strip()
if not API_TOKEN:
    raise RuntimeError(
        "FACE_API_TOKEN is not set. Refusing to start an unauthenticated "
        "face recognition service that listens on 0.0.0.0:8000. Set the "
        "FACE_API_TOKEN environment variable (see docker-compose.yml)."
    )

app = FastAPI(title="Face Recognition Encoder API", version="2.0.0")


def verify_token(authorization: Optional[str] = Header(None)) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization[len("Bearer "):]
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def _load_image(raw: bytes) -> np.ndarray:
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image upload")
    try:
        return face_recognition.load_image_file(io.BytesIO(raw))
    except Exception as err:  # noqa: BLE001 - surfaced to the caller as 400
        raise HTTPException(status_code=400, detail=f"Invalid image: {err}") from err


def _clamp_jitters(num_jitters: int) -> int:
    return max(0, min(int(num_jitters), MAX_JITTERS))


@app.get("/health")
def health(_: None = Depends(verify_token)) -> dict:
    return {
        "status": "ok",
        "engine": ENGINE_ID,
        "encoding_size": ENCODING_SIZE,
        "detector_model": "hog",
    }


@app.post("/v1/encode/single")
async def encode_single(
    image: UploadFile = File(...),
    num_jitters: int = Form(DEFAULT_ENROLL_JITTERS),
    _: None = Depends(verify_token),
) -> dict:
    """Encode exactly one face, for enrolling a reference photo.

    Fails with 422/no_face or 422/multiple_faces if the photo doesn't contain
    exactly one detectable face, mirroring the local dlib worker's enrollment
    contract in the HA integration.
    """

    raw = await image.read()
    img = _load_image(raw)
    locations = face_recognition.face_locations(img)

    if len(locations) == 0:
        raise HTTPException(
            status_code=422,
            detail={"error_code": "no_face", "message": "На фотографии не найдено лицо"},
        )
    if len(locations) > 1:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "multiple_faces",
                "faces_detected": len(locations),
                "message": (
                    f"Для эталона требуется ровно одно лицо; найдено: {len(locations)}"
                ),
            },
        )

    encodings = face_recognition.face_encodings(
        img, known_face_locations=locations, num_jitters=_clamp_jitters(num_jitters)
    )
    return {"encoding": [float(v) for v in encodings[0]], "faces_detected": 1}


@app.post("/v1/encode/multi")
async def encode_multi(
    image: UploadFile = File(...),
    num_jitters: int = Form(DEFAULT_RECOGNIZE_JITTERS),
    _: None = Depends(verify_token),
) -> dict:
    """Detect and encode every face in a frame, for a recognition pass."""

    raw = await image.read()
    img = _load_image(raw)
    locations = face_recognition.face_locations(img)
    encodings = face_recognition.face_encodings(
        img, known_face_locations=locations, num_jitters=_clamp_jitters(num_jitters)
    )

    faces = [
        {"encoding": [float(v) for v in enc], "bbox": [int(c) for c in loc]}
        for enc, loc in zip(encodings, locations)
    ]
    return {"faces_detected": len(faces), "faces": faces}
