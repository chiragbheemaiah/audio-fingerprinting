from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fingerprint import predict

app = FastAPI()
templates = Jinja2Templates(directory="pages")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_BYTES = 50 * 1024 * 1024  # 50 MB

QUERY_DIR = Path("query")
QUERY_DIR.mkdir(parents=True, exist_ok=True)

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

    results = predict()
    print("Results: ", results)
    print("Empty query folder")
    try:
        if dest.is_file() or dest.is_symlink():
            dest.unlink() # Delete the file
            print(f"Deleted: {dest}")
    except OSError as e:
        print(f"Error deleting {dest}: {e.strerror}")
        
    return {
        "results": results
    }

    