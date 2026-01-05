import os
import pickle
from fastapi import FastAPI, UploadFile, File, HTTPException
import face_recognition
from PIL import Image
import numpy as np

BASE_DIR = "/app"
FACES_DIR = "/app/faces"
ENCODINGS_FILE = "/app/faces/encodings.pkl"

app = FastAPI(title="Face Recognition API")

os.makedirs(FACES_DIR, exist_ok=True)

# -------------------------
# load / save
# -------------------------

def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_encodings(data):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

encodings_db = load_encodings()

# -------------------------
# helpers
# -------------------------

def encode_image(file) -> np.ndarray:
    image = face_recognition.load_image_file(file)
    enc = face_recognition.face_encodings(image)
    if not enc:
        raise ValueError("Face not found")
    return enc[0]

def rebuild_person(name):
    person_dir = os.path.join(FACES_DIR, name)
    person_encodings = []

    for img in os.listdir(person_dir):
        path = os.path.join(person_dir, img)
        try:
            e = encode_image(path)
            person_encodings.append(e)
        except:
            continue

    if not person_encodings:
        raise ValueError("No valid faces")

    encodings_db[name] = person_encodings
    save_encodings(encodings_db)

# -------------------------
# API
# -------------------------

@app.get("/persons")
def persons():
    return list(encodings_db.keys())

@app.post("/person")
def add_person(name: str):
    path = os.path.join(FACES_DIR, name)
    if os.path.exists(path):
        raise HTTPException(400, "Person exists")
    os.makedirs(path)
    encodings_db[name] = []
    save_encodings(encodings_db)
    return {"status": "created", "name": name}

@app.post("/person/{name}/photo")
async def add_photo(name: str, file: UploadFile = File(...)):
    if name not in encodings_db:
        raise HTTPException(404, "Person not found")

    person_dir = os.path.join(FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    img_path = os.path.join(person_dir, file.filename)
    with open(img_path, "wb") as f:
        f.write(await file.read())

    try:
        rebuild_person(name)
    except ValueError as e:
        os.remove(img_path)
        raise HTTPException(400, str(e))

    return {"status": "photo added", "name": name}

@app.delete("/person/{name}")
def delete_person(name: str):
    if name not in encodings_db:
        raise HTTPException(404)

    import shutil
    shutil.rmtree(os.path.join(FACES_DIR, name))
    encodings_db.pop(name)
    save_encodings(encodings_db)

    return {"status": "deleted"}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    image = face_recognition.load_image_file(file.file)
    
    locations = face_recognition.face_locations(image)
    encs = face_recognition.face_encodings(image, locations)

    results = []

    for idx, test_enc in enumerate(encs):
        found = "unknown"

        for name, known_encs in encodings_db.items():
            matches = face_recognition.compare_faces(
                known_encs,
                test_enc,
                tolerance=0.6
            )
            if True in matches:
                found = name
                break

        results.append({
            "id": idx,
            "name": found
        })

    return {
        "count": len(results),
        "faces": results
    }
    
    