# Identifying Audio Snippets by Their Signatures

This project implements an audio fingerprinting system that identifies short audio clips by matching their spectral signatures against a stored reference library. It combines signal processing techniques with a lightweight web interface built using FastAPI and Bootstrap.

---

## Overview

The system allows users to:

* Upload reference audio files to build a searchable library.
* Submit query clips for identification.
* Compare query fingerprints with stored fingerprints to find the closest match.

Each audio file is converted into a mel-spectrogram, from which key peaks are extracted to form a constellation map. Fuzzy hash functions are then used to generate compact, noise-tolerant fingerprints that remain consistent under minor distortions in time or frequency.

---

## Core Components

### 1. Endpoints (FastAPI)

* **`/upload`** – Accepts and stores reference audio files.
* **`/predict`** – Processes a query file and compares it with stored fingerprints.
* **`/`** – Renders the homepage with a list of uploaded files.

The backend uses `librosa` for audio analysis and fingerprint generation.

### 2. Fingerprinting Engine

* Reads mono audio samples using `librosa.load()`.
* Computes mel-spectrograms.
* Identifies local maxima in time-frequency windows (constellations).
* Generates fuzzy hashes based on frequency and time pairings for efficient lookup and matching.

---

## Running the Application

1. **Install dependencies**

   ```bash
   pip install fastapi uvicorn librosa matplotlib jinja2
   ```

2. **Start the server**

   ```bash
   uvicorn server:app --reload
   ```

3. **Access the web interface**
   Open your browser and go to:
   `http://localhost:8000`

---

## Project Structure

```
project/
│
├── server.py              # FastAPI application
├── templates/
│   └── home.html          # Jinja2 frontend template
├── uploads/               # Stored reference audio
├── query/                 # Query audio files
└── README.md
```

---

## References

* Avery Li-Chun Wang, *An Industrial-Strength Audio Search Algorithm (Shazam)*
