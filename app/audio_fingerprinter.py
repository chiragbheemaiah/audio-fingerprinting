"""
Audio Fingerprinting Library

A modular, reusable audio fingerprinting system based on spectral peak matching.
Inspired by the Shazam algorithm.
"""

import numpy as np
import librosa
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Dict
from pathlib import Path


class AudioFingerprinter:
    """
    Core audio fingerprinting algorithm.
    
    Extracts acoustic fingerprints from audio files using mel spectrograms
    and constellation-based peak detection.
    """
    
    # Fuzzy hash parameters (Shazam-style)
    FUZ_F_HZ = 20.0    # frequency fuzz (≈ FFT/mel bin spacing)
    FUZ_DT_S = 0.02    # time fuzz (≈ hop size in seconds)
    
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        window_size: int = 15,
        threshold: float = 0.2,
        time_window: float = 5.0
    ):
        """
        Initialize the fingerprinter with signal processing parameters.
        
        Args:
            n_fft: FFT window size
            hop_length: Number of samples between frames
            n_mels: Number of mel frequency bins
            window_size: Size of window for peak detection
            threshold: Fraction of lowest peaks to discard
            time_window: Maximum time difference for peak pairing (seconds)
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.window_size = window_size
        self.threshold = threshold
        self.time_window = time_window
    
    def read_audio(self, filepath: str) -> Optional[Tuple[int, np.ndarray]]:
        """
        Load audio file as mono float32.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Tuple of (sample_rate, audio_waveform) or (None, None) on error
        """
        if not Path(filepath).exists():
            return None, None
        
        try:
            audio, sr = librosa.load(filepath, sr=None, mono=True)
            return sr, audio.astype(np.float32)
        except Exception as e:
            print(f"Error reading audio file: {e}")
            return None, None
    
    def compute_spectrogram(
        self, 
        audio: np.ndarray, 
        sr: int,
        power: float = 1.0
    ) -> np.ndarray:
        """
        Compute mel spectrogram.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            power: Exponent for magnitude spectrogram (1.0=magnitude, 2.0=power)
            
        Returns:
            Mel spectrogram with shape (n_mels, time_frames)
        """
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=power
        )
        return S
    
    def extract_peaks(
        self, 
        spectrogram: np.ndarray
    ) -> List[List[int]]:
        """
        Extract constellation peaks from spectrogram.
        
        Divides spectrogram into windows and finds local maxima, then
        filters out low-amplitude peaks.
        
        Args:
            spectrogram: Mel spectrogram array
            
        Returns:
            List of peaks as [freq_idx, time_idx, amplitude]
        """
        max_points = []
        r, c = spectrogram.shape
        
        # Find local maxima in each window
        for i in range(0, r, self.window_size):
            for j in range(0, c, self.window_size):
                max_ele = -1
                max_k, max_l = -1, -1
                
                for k in range(i, min(i + self.window_size, r)):
                    for l in range(j, min(j + self.window_size, c)):
                        if spectrogram[k, l] > max_ele:
                            max_ele = spectrogram[k, l]
                            max_k, max_l = k, l
                
                if max_k >= 0 and max_l >= 0:
                    max_points.append([max_k, max_l, spectrogram[max_k, max_l]])
        
        # Filter out low-amplitude peaks
        sorted_points = sorted(max_points, key=lambda c: c[2])
        filter_idx = int(len(sorted_points) * self.threshold)
        return sorted_points[filter_idx:]
    
    def _make_fuzzy_hash(
        self, 
        f1_hz: float, 
        f2_hz: float, 
        dt_s: float
    ) -> int:
        """
        Create fuzzy composite hash from frequency pair and time difference.
        
        Quantizes values to coarse buckets for noise tolerance.
        
        Args:
            f1_hz: First frequency in Hz
            f2_hz: Second frequency in Hz
            dt_s: Time difference in seconds
            
        Returns:
            Integer hash
        """
        f1_q = f1_hz - (f1_hz % self.FUZ_F_HZ)
        f2_q = f2_hz - (f2_hz % self.FUZ_F_HZ)
        dt_q = dt_s - (dt_s % self.FUZ_DT_S)
        
        b1 = int(f1_q // self.FUZ_F_HZ)
        b2 = int(f2_q // self.FUZ_F_HZ)
        b3 = int(dt_q // self.FUZ_DT_S)
        
        return (b3 * 10**6) + (b2 * 10**3) + b1
    
    def generate_fingerprints(
        self,
        peaks: List[List],
        times: np.ndarray,
        freqs: np.ndarray,
        audio_id: int
    ) -> List[Tuple[int, int, float]]:
        """
        Generate fingerprint hashes from constellation peaks.
        
        Creates hashes from pairs of peaks within the time window.
        
        Args:
            peaks: List of [freq_idx, time_idx, amplitude]
            times: Time values for each frame
            freqs: Frequency values for each mel bin
            audio_id: Identifier for this audio
            
        Returns:
            List of (hash_digest, audio_id, anchor_time) tuples
        """
        hashes = []
        
        for x, (xi, xj, _) in enumerate(peaks):
            for y, (yi, yj, _) in enumerate(peaks):
                if x != y:
                    time_x = times[xj]
                    time_y = times[yj]
                    freq_x = freqs[xi]
                    freq_y = freqs[yi]
                    time_diff = time_y - time_x
                    
                    if 0 < time_diff <= self.time_window:
                        digest = self._make_fuzzy_hash(freq_x, freq_y, time_diff)
                        hashes.append((digest, audio_id, time_x))
        
        return hashes
    
    def process(
        self, 
        filepath: str, 
        audio_id: int
    ) -> Optional[List[Tuple[int, int, float]]]:
        """
        Complete fingerprinting pipeline for an audio file.
        
        Args:
            filepath: Path to audio file
            audio_id: Identifier for this audio
            
        Returns:
            List of fingerprint hashes or None on error
        """
        # Read audio
        sr, audio = self.read_audio(filepath)
        if sr is None or audio is None:
            return None
        
        # Compute spectrogram
        S = self.compute_spectrogram(audio, sr, power=1.0)
        
        # Extract peaks
        peaks = self.extract_peaks(S)
        
        # Generate time and frequency arrays
        times = librosa.frames_to_time(
            np.arange(S.shape[1]), 
            sr=sr, 
            hop_length=self.hop_length
        )
        freqs = librosa.mel_frequencies(
            S.shape[0], 
            fmin=0, 
            fmax=sr/2
        )
        
        # Generate fingerprints
        hashes = self.generate_fingerprints(peaks, times, freqs, audio_id)
        
        return hashes


class FingerprintDatabase:
    """
    Storage and matching for audio fingerprints.
    """
    
    def __init__(self):
        """Initialize empty database."""
        self.fingerprints = defaultdict(list)  # hash -> [(audio_id, time), ...]
        self.metadata = {}  # audio_id -> metadata dict
    
    def add_fingerprints(
        self,
        hashes: List[Tuple[int, int, float]],
        audio_id: int,
        metadata: Optional[Dict] = None
    ):
        """
        Add fingerprints to database.
        
        Args:
            hashes: List of (hash_digest, audio_id, time) tuples
            audio_id: Identifier for this audio
            metadata: Optional metadata dict (e.g., {"filename": "song.wav"})
        """
        for digest, aid, time_x in hashes:
            self.fingerprints[digest].append((aid, time_x))
        
        if metadata:
            self.metadata[audio_id] = metadata
    
    def query(
        self,
        query_hashes: List[Tuple[int, int, float]]
    ) -> Optional[List[Tuple[int, float]]]:
        """
        Find best matches for query fingerprints.
        
        Args:
            query_hashes: List of fingerprint hashes from query audio
            
        Returns:
            List of (audio_id, confidence) tuples sorted by confidence,
            or None if no matches found
        """
        matches = defaultdict(list)
        
        # Find matching hashes and compute time offsets
        for digest, _, time_x in query_hashes:
            if digest in self.fingerprints:
                for audio_id, anchor_time_db in self.fingerprints[digest]:
                    offset = int(anchor_time_db - time_x)
                    matches[offset].append(audio_id)
        
        if not matches:
            return None
        
        # Find offset with most matches
        max_key = max(matches.keys(), key=lambda k: len(matches[k]))
        
        # Count audio_id occurrences at best offset
        counter = Counter(matches[max_key])
        total = sum(counter.values())
        
        # Return sorted by confidence
        results = [
            (audio_id, count / total)
            for audio_id, count in counter.items()
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def clear(self):
        """Clear all fingerprints and metadata."""
        self.fingerprints.clear()
        self.metadata.clear()
    
    def get_metadata(self, audio_id: int) -> Optional[Dict]:
        """Get metadata for an audio ID."""
        return self.metadata.get(audio_id)
