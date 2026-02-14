from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from audio_fingerprinter import AudioFingerprinter, FingerprintDatabase

app = FastAPI()
templates = Jinja2Templates(directory="pages")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_BYTES = 50 * 1024 * 1024  # 50 MB

QUERY_DIR = Path("query")
QUERY_DIR.mkdir(parents=True, exist_ok=True)

# Initialize fingerprinter and database
fingerprinter = AudioFingerprinter()
database = FingerprintDatabase()

@app.get("/")
def home(request: Request):
    files = sorted(
        [f for f in UPLOAD_DIR.iterdir() if f.is_file()],
        key=lambda p: p.name.lower()
    )
    # build simple structures the template can use
    file_items = [{"name": f.name, "url": f"/uploads/{f.name}"} for f in files]
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "title": "Audio Fingerprinting",
            "uploaded_files": file_items,
        },
    )

@app.post("/upload")
async def upload_audio(audio: UploadFile = File(...)):
    
    filename = audio.filename or "unnamed_audio_file"

    dest = UPLOAD_DIR / filename
    total = 0

    with dest.open("wb") as f:
        while True:
            chunk = await audio.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_BYTES:
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large")
            f.write(chunk)

    return {"filename": filename, "bytes": total}

@app.post("/predict")
async def process_query(audio: UploadFile = File(...)):
    filename = audio.filename or "unnamed_audio_file"

    dest = QUERY_DIR / filename
    total = 0

    with dest.open("wb") as f:
        while True:
            chunk = await audio.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_BYTES:
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large")
            f.write(chunk)

    # Clear database and reprocess all reference files
    database.clear()
    
    # Process all uploaded reference files
    audio_id = 0
    for ref_file in UPLOAD_DIR.iterdir():
        if ref_file.is_file():
            audio_id += 1
            hashes = fingerprinter.process(str(ref_file), audio_id)
            if hashes:
                database.add_fingerprints(
                    hashes, 
                    audio_id, 
                    metadata={"filename": ref_file.name}
                )
    
    # Process query file
    query_hashes = fingerprinter.process(str(dest), 999)
    
    # Clean up query file
    try:
        if dest.is_file() or dest.is_symlink():
            dest.unlink()
    except OSError as e:
        print(f"Error deleting {dest}: {e.strerror}")
    
    if not query_hashes:
        return {"results": []}
    
    # Find matches
    results = database.query(query_hashes)
    
    if results is None:
        return {"results": []}
    
    # Format response
    response = []
    for audio_id, confidence in results:
        metadata = database.get_metadata(audio_id)
        response.append({
            "audio_id": audio_id,
            "label": metadata.get("filename", f"audio_{audio_id}") if metadata else f"audio_{audio_id}",
            "probability": confidence
        })
    
    return {"results": response}