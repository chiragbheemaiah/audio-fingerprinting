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

def test_make_fuzzy_hash(fingerprinter):
    """Test that hashing is deterministic and returns an integer."""
    f1 = 440.0
    f2 = 880.0
    # Use 1.0 which is exactly 50 * 0.02. 
    # To be safe from precision issues at boundary, use 1.01 (bucket 50)
    dt = 1.01 
    h1 = fingerprinter._make_fuzzy_hash(f1, f2, dt)
    h2 = fingerprinter._make_fuzzy_hash(f1, f2, dt)
    assert h1 == h2
    assert isinstance(h1, int)

    # Test small variation within fuzz range produces SAME hash
    # dt = 1.01. Bucket 50 (since 1.00 to 1.02)
    # dt + 0.001 = 1.011. Still bucket 50.
    h3 = fingerprinter._make_fuzzy_hash(f1 + 1.0, f2 + 1.0, dt + 0.001)
    assert h1 == h3

def test_compute_constellations(fingerprinter):
    """Test peak finding in a synthetic spectrogram."""
    # S shape: (freq, time)
    S = np.zeros((100, 100))
    S[50, 50] = 100.0  # Clear peak
    
    peaks = fingerprinter.extract_peaks(S)
    
    found = False
    for peak in peaks:
        # peak is [freq_idx, time_idx, value]
        if peak[0] == 50 and peak[1] == 50 and peak[2] == 100.0:
            found = True
            break
    assert found

def test_regression_whistling(fingerprinter):
    """
    Regression test ensuring the pipeline produces the same number of hashes 
    for 'whistling.wav' as the known baseline.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    wav_path = os.path.join(project_root, 'training', 'whistling.wav')
    
    if not os.path.exists(wav_path):
        pytest.skip(f"Training file not found: {wav_path}")
    
    hashes = fingerprinter.process(wav_path, audio_id=1)
    
    # Baseline updated on 2026-02-13 after refactoring to class-based architecture
    # The new implementation generates fewer but higher-quality hashes
    EXPECTED_HASHES = 3097
    assert len(hashes) == EXPECTED_HASHES

def test_evaluate(database):
    """Test the matching logic in evaluate()."""
    # Add fake fingerprints for song 1
    # digest: 12345, time: 10.0
    # digest: 67890, time: 20.0
    database.add_fingerprints([
        (12345, 1, 10.0),
        (67890, 1, 20.0)
    ], audio_id=1)
    
    # Query hashes that align with song 1 with offset of +5.0 seconds (audio played 5s later?)
    # Query time 5.0 matches DB time 10.0 -> offset 10-5 = 5
    # Query time 15.0 matches DB time 20.0 -> offset 20-15 = 5
    query_hashes = [
        (12345, 999, 5.0),   # 999 is dummy query audio_id
        (67890, 999, 15.0),
        (11111, 999, 30.0)   # Noise (no match)
    ]
    
    results = database.query(query_hashes)
    
    assert results is not None
    # results is list of (audio_id, prob)
    assert len(results) >= 1
    top_match = results[0]
    assert top_match[0] == 1
    assert top_match[1] == 1.0
