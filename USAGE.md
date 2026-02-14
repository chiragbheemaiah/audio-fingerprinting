# Audio Fingerprinter API

A modular, reusable audio fingerprinting library for identifying audio clips.

## Installation

```bash
uv sync
```

## Quick Start

```python
from audio_fingerprinter import AudioFingerprinter, FingerprintDatabase

# Initialize
fingerprinter = AudioFingerprinter()
database = FingerprintDatabase()

# Add reference audio
audio_id = 1
hashes = fingerprinter.process("reference_song.wav", audio_id)
database.add_fingerprints(hashes, audio_id, metadata={"title": "My Song"})

# Query with audio clip
query_hashes = fingerprinter.process("clip.wav", audio_id=999)
results = database.query(query_hashes)

if results:
    for audio_id, confidence in results:
        metadata = database.get_metadata(audio_id)
        print(f"Match: {metadata['title']} ({confidence:.2%} confident)")
```

## API Reference

### AudioFingerprinter

**Constructor:**
```python
AudioFingerprinter(
    n_fft=2048,          # FFT window size
    hop_length=512,      # Samples between frames
    n_mels=128,          # Mel frequency bins
    window_size=15,      # Peak detection window
    threshold=0.2,       # Low-peak filter threshold
    time_window=5.0      # Max time for peak pairing (seconds)
)
```

**Methods:**
- `process(filepath, audio_id)` → List of fingerprint hashes
- `read_audio(filepath)` → (sample_rate, audio_array)
- `compute_spectrogram(audio, sr)` → mel spectrogram
- `extract_peaks(spectrogram)` → constellation peaks
- `generate_fingerprints(peaks, times, freqs, audio_id)` → hashes

### FingerprintDatabase

**Methods:**
- `add_fingerprints(hashes, audio_id, metadata=None)` - Store fingerprints
- `query(query_hashes)` → List of (audio_id, confidence) or None
- `get_metadata(audio_id)` → metadata dict
- `clear()` - Reset database

## Running the Web Server

```bash
cd app
uv run uvicorn server:app --reload
```

Visit `http://localhost:8000` to use the web interface.

## Running Tests

```bash
uv run pytest
```

All 11 tests should pass.
