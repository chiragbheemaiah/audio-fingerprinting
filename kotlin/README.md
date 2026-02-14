# Audio Fingerprinting for Android

Kotlin implementation of the audio fingerprinting algorithm for Android integration.

## Integration Guide

### 1. Add to Your Android Project

Copy the `kotlin/src/main/kotlin/com/audiofinger` folder to your Android project:

```
YourApp/app/src/main/kotlin/com/audiofinger/
└── AudioFingerprinter.kt
```

### 2. Dependencies

Add to your `build.gradle.kts`:

```kotlin
dependencies {
    // For audio processing, you'll need an audio library
    // Recommended: TarsosDSP or custom FFT implementation
}
```

### 3. Basic Usage

```kotlin
import com.audiofinger.AudioFingerprinter
import com.audiofinger.FingerprintDatabase

// Initialize
val fingerprinter = AudioFingerprinter()
val database = FingerprintDatabase()

// Process reference audio
// Note: You need to provide mel spectrogram, times, and freqs arrays
// (these come from audio processing library like TarsosDSP)
val peaks = fingerprinter.extractPeaks(melSpectrogram)
val hashes = fingerprinter.generateFingerprints(peaks, times, freqs, audioId = 1)
database.addFingerprints(hashes, audioId = 1, mapOf("title" to "Song Name"))

// Query with audio clip
val queryPeaks = fingerprinter.extractPeaks(querySpectrogram)
val queryHashes = fingerprinter.generateFingerprints(queryPeaks, queryTimes, queryFreqs, 999)
val results = database.query(queryHashes)

results?.forEach { match ->
    val meta = database.getMetadata(match.audioId)
    println("Match: ${meta?.get("title")} (${match.confidence * 100}% confident)")
}
```

## Audio Processing

The Kotlin implementation focuses on the fingerprinting algorithm. You need to provide:

1. **Mel Spectrogram**: 2D array `[freq, time]` from audio analysis
2. **Times Array**: Time values for each frame
3. **Freqs Array**: Frequency values for each mel bin

### Recommended Libraries

**Option 1: TarsosDSP**
```kotlin
// Add dependency
implementation("be.tarsos.dsp:core:2.5")

// Use for FFT and mel spectrogram
```

**Option 2: Use Python Backend**
```kotlin
// Call Python microservice for audio processing
// Send audio file, receive spectrogram arrays
```

## Example Android Implementation

```kotlin
class AudioFingerprintService(context: Context) {
    private val fingerprinter = AudioFingerprinter()
    private val database = FingerprintDatabase()
    
    suspend fun addReferenceAudio(audioFile: File, audioId: Int) = withContext(Dispatchers.IO) {
        // 1. Load audio file
        val audioData = loadAudioFile(audioFile)
        
        // 2. Compute mel spectrogram (use audio processing library)
        val (spectrogram, times, freqs) = computeMelSpectrogram(audioData)
        
        // 3. Extract peaks and generate fingerprints
        val peaks = fingerprinter.extractPeaks(spectrogram)
        val hashes = fingerprinter.generateFingerprints(peaks, times, freqs, audioId)
        
        // 4. Store in database
        database.addFingerprints(
            hashes, 
            audioId, 
            mapOf("filename" to audioFile.name)
        )
    }
    
    suspend fun identifyAudio(audioClip: File): List<MatchResult>? = withContext(Dispatchers.IO) {
        val audioData = loadAudioFile(audioClip)
        val (spectrogram, times, freqs) = computeMelSpectrogram(audioData)
        val peaks = fingerprinter.extractPeaks(spectrogram)
        val hashes = fingerprinter.generateFingerprints(peaks, times, freqs, 999)
        database.query(hashes)
    }
}
```

## Configuration

Customize fingerprinter parameters:

```kotlin
val fingerprinter = AudioFingerprinter(
    nFft = 2048,         // FFT window size
    hopLength = 512,     // Samples between frames
    nMels = 128,         // Mel frequency bins
    windowSize = 15,     // Peak detection window
    threshold = 0.2f,    // Low-peak filter (0-1)
    timeWindow = 5.0f    // Max time for pairing (seconds)
)
```

## Performance Considerations

- **Background Processing**: Run fingerprinting on background threads
- **Caching**: Store processed fingerprints in local database (Room/SQLite)
- **Memory**: Large spectrograms can use significant memory
- **Battery**: Audio processing is CPU-intensive

## Integration Checklist

- [ ] Copy `AudioFingerprinter.kt` to your project
- [ ] Add audio processing library or backend service
- [ ] Implement mel spectrogram computation
- [ ] Test with sample audio files
- [ ] Add proper error handling
- [ ] Consider using Room for persistent storage
- [ ] Implement background workers for processing

## Support

For the full Python reference implementation and algorithm details, see the parent directory.
