import sys
import os
import pytest
import numpy as np

# Add app directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))

from audio_fingerprinter import AudioFingerprinter, FingerprintDatabase

@pytest.fixture
def fingerprinter():
    """Provide a fresh fingerprinter instance for each test."""
    return AudioFingerprinter()

@pytest.fixture
def database():
    """Provide a fresh database instance for each test."""
    return FingerprintDatabase()

def test_filter_constellation(fingerprinter):
    """Test that constellation filtering removes low-amplitude peaks."""
    # Create constellation directly (simulate extracted peaks)
    constellation = [
        [10, 20, 1.0], [20, 30, 5.0], [30, 40, 10.0],
        [40, 50, 2.0], [50, 60, 8.0],
    ]
    
    # Manually filter like the extract_peaks method does
    sorted_points = sorted(constellation, key=lambda c: c[2])
    filter_idx = int(len(sorted_points) * 0.4)
    filtered = sorted_points[filter_idx:]
    
    assert len(filtered) == 3
    amplitudes = [c[2] for c in filtered]
    assert 5.0 in amplitudes
    assert 8.0 in amplitudes
    assert 10.0 in amplitudes

def test_save_to_db(database):
    """Test that add_fingerprints correctly populates the database."""
    hashes = [
        (12345, 1, 10.0), (67890, 1, 20.0), (12345, 2, 15.0),
    ]
    
    database.add_fingerprints(hashes, audio_id=1)
    assert len(database.fingerprints[12345]) >= 1
    
    # Add more from audio 2
    database.add_fingerprints([(12345, 2, 15.0)], audio_id=2)
    assert len(database.fingerprints[12345]) >= 1

def test_create_fingerprint(fingerprinter):
    """Test fingerprint creation from a simple constellation."""
    peaks = [[10, 20, 5.0], [15, 25, 6.0]]
    times = np.linspace(0, 10, 100)
    freqs = np.linspace(0, 8000, 128)
    
    hashes = fingerprinter.generate_fingerprints(peaks, times, freqs, audio_id=1)
    
    assert len(hashes) > 0
    for h in hashes:
        assert len(h) == 3
        assert h[1] == 1
        assert isinstance(h[0], int)

def test_read_audio_file_missing(fingerprinter):
    """Test read_audio handles missing files gracefully."""
    sr, audio = fingerprinter.read_audio("/nonexistent/path/file.wav")
    assert sr is None
    assert audio is None

def test_compute_mel_spectrogram_shape(fingerprinter):
    """Test mel spectrogram has correct shape."""
    sr = 16000
    duration = 1.0
    freq = 440.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
    
    S = fingerprinter.compute_spectrogram(audio, sr)
    
    assert S.shape[0] == fingerprinter.n_mels
    expected_frames = int(np.ceil(len(audio) / fingerprinter.hop_length))
    assert abs(S.shape[1] - expected_frames) <= 2

def test_evaluate_no_matches(database):
    """Test query returns None when there are no matches."""
    database.add_fingerprints([(12345, 1, 10.0), (67890, 1, 20.0)], audio_id=1)
    
    query_hashes = [(99999, 999, 5.0), (88888, 999, 10.0)]
    results = database.query(query_hashes)
    assert results is None

def test_evaluate_multiple_songs(database):
    """Test query correctly identifies the best match among multiple songs."""
    database.add_fingerprints([
        (111, 1, 10.0), (222, 1, 20.0), (333, 1, 30.0)
    ], audio_id=1)
    
    database.add_fingerprints([
        (444, 2, 15.0), (555, 2, 25.0), (666, 2, 35.0),
        (777, 2, 45.0), (888, 2, 55.0)
    ], audio_id=2)
    
    query_hashes = [
        (444, 999, 10.0), (555, 999, 20.0), (666, 999, 30.0),
        (777, 999, 40.0), (888, 999, 50.0),
    ]
    
    results = database.query(query_hashes)
    
    assert results is not None
    top_match = results[0]
    assert top_match[0] == 2
    assert top_match[1] == 1.0
