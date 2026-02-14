package com.audiofinger

import kotlin.math.ceil
import kotlin.math.floor

/**
 * Audio Fingerprinting Library for Android
 * 
 * A Kotlin implementation of the audio fingerprinting algorithm
 * based on spectral peak constellation matching (Shazam-like).
 */

/**
 * Core audio fingerprinting algorithm.
 * 
 * Extracts acoustic fingerprints from audio samples using mel spectrograms
 * and constellation-based peak detection.
 * 
 * @property nFft FFT window size
 * @property hopLength Number of samples between frames
 * @property nMels Number of mel frequency bins
 * @property windowSize Size of window for peak detection
 * @property threshold Fraction of lowest peaks to discard
 * @property timeWindow Maximum time difference for peak pairing (seconds)
 */
class AudioFingerprinter(
    private val nFft: Int = 2048,
    private val hopLength: Int = 512,
    private val nMels: Int = 128,
    private val windowSize: Int = 15,
    private val threshold: Float = 0.2f,
    private val timeWindow: Float = 5.0f
) {
    companion object {
        // Fuzzy hash parameters (Shazam-style)
        private const val FUZ_F_HZ = 20.0f  // frequency fuzz
        private const val FUZ_DT_S = 0.02f  // time fuzz
    }

    /**
     * Data class representing a spectral peak
     */
    data class Peak(
        val freqIdx: Int,
        val timeIdx: Int,
        val amplitude: Float
    )

    /**
     * Data class representing a fingerprint hash
     */
    data class Fingerprint(
        val digest: Long,
        val audioId: Int,
        val anchorTime: Float
    )

    /**
     * Extract constellation peaks from spectrogram.
     * 
     * Divides spectrogram into windows and finds local maxima,
     * then filters out low-amplitude peaks.
     * 
     * @param spectrogram Mel spectrogram array [freq, time]
     * @return List of peaks
     */
    fun extractPeaks(spectrogram: Array<FloatArray>): List<Peak> {
        val numFreqs = spectrogram.size
        val numTimes = spectrogram[0].size
        val peaks = mutableListOf<Peak>()

        // Find local maxima in each window
        for (i in 0 until numFreqs step windowSize) {
            for (j in 0 until numTimes step windowSize) {
                var maxValue = -1f
                var maxFreqIdx = -1
                var maxTimeIdx = -1

                for (k in i until minOf(i + windowSize, numFreqs)) {
                    for (l in j until minOf(j + windowSize, numTimes)) {
                        if (spectrogram[k][l] > maxValue) {
                            maxValue = spectrogram[k][l]
                            maxFreqIdx = k
                            maxTimeIdx = l
                        }
                    }
                }

                if (maxFreqIdx >= 0 && maxTimeIdx >= 0) {
                    peaks.add(Peak(maxFreqIdx, maxTimeIdx, maxValue))
                }
            }
        }

        // Filter out low-amplitude peaks
        val sortedPeaks = peaks.sortedBy { it.amplitude }
        val filterIdx = (sortedPeaks.size * threshold).toInt()
        return sortedPeaks.subList(filterIdx, sortedPeaks.size)
    }

    /**
     * Create fuzzy composite hash from frequency pair and time difference.
     * 
     * Quantizes values to coarse buckets for noise tolerance.
     * 
     * @param f1Hz First frequency in Hz
     * @param f2Hz Second frequency in Hz
     * @param dtS Time difference in seconds
     * @return Integer hash
     */
    private fun makeFuzzyHash(f1Hz: Float, f2Hz: Float, dtS: Float): Long {
        val f1Q = f1Hz - (f1Hz % FUZ_F_HZ)
        val f2Q = f2Hz - (f2Hz % FUZ_F_HZ)
        val dtQ = dtS - (dtS % FUZ_DT_S)

        val b1 = floor(f1Q / FUZ_F_HZ).toLong()
        val b2 = floor(f2Q / FUZ_F_HZ).toLong()
        val b3 = floor(dtQ / FUZ_DT_S).toLong()

        return (b3 * 1_000_000) + (b2 * 1_000) + b1
    }

    /**
     * Generate fingerprint hashes from constellation peaks.
     * 
     * Creates hashes from pairs of peaks within the time window.
     * 
     * @param peaks List of spectral peaks
     * @param times Time values for each frame
     * @param freqs Frequency values for each mel bin
     * @param audioId Identifier for this audio
     * @return List of fingerprint hashes
     */
    fun generateFingerprints(
        peaks: List<Peak>,
        times: FloatArray,
        freqs: FloatArray,
        audioId: Int
    ): List<Fingerprint> {
        val hashes = mutableListOf<Fingerprint>()

        for (i in peaks.indices) {
            for (j in peaks.indices) {
                if (i != j) {
                    val peak1 = peaks[i]
                    val peak2 = peaks[j]

                    val time1 = times[peak1.timeIdx]
                    val time2 = times[peak2.timeIdx]
                    val freq1 = freqs[peak1.freqIdx]
                    val freq2 = freqs[peak2.freqIdx]
                    val timeDiff = time2 - time1

                    if (timeDiff in 0f..timeWindow) {
                        val digest = makeFuzzyHash(freq1, freq2, timeDiff)
                        hashes.add(Fingerprint(digest, audioId, time1))
                    }
                }
            }
        }

        return hashes
    }
}


/**
 * Storage and matching for audio fingerprints.
 */
class FingerprintDatabase {
    private val fingerprints = mutableMapOf<Long, MutableList<Pair<Int, Float>>>()
    private val metadata = mutableMapOf<Int, Map<String, String>>()

    /**
     * Data class for match results
     */
    data class MatchResult(
        val audioId: Int,
        val confidence: Float
    )

    /**
     * Add fingerprints to database.
     * 
     * @param hashes List of fingerprint hashes
     * @param audioId Identifier for this audio
     * @param meta Optional metadata map
     */
    fun addFingerprints(
        hashes: List<AudioFingerprinter.Fingerprint>,
        audioId: Int,
        meta: Map<String, String>? = null
    ) {
        for (hash in hashes) {
            fingerprints.getOrPut(hash.digest) { mutableListOf() }
                .add(Pair(hash.audioId, hash.anchorTime))
        }

        meta?.let { metadata[audioId] = it }
    }

    /**
     * Find best matches for query fingerprints.
     * 
     * @param queryHashes List of fingerprint hashes from query audio
     * @return List of match results sorted by confidence, or null if no matches
     */
    fun query(queryHashes: List<AudioFingerprinter.Fingerprint>): List<MatchResult>? {
        val matches = mutableMapOf<Int, MutableList<Int>>()

        // Find matching hashes and compute time offsets
        for (queryHash in queryHashes) {
            fingerprints[queryHash.digest]?.forEach { (audioId, anchorTimeDb) ->
                val offset = (anchorTimeDb - queryHash.anchorTime).toInt()
                matches.getOrPut(offset) { mutableListOf() }.add(audioId)
            }
        }

        if (matches.isEmpty()) return null

        // Find offset with most matches
        val maxOffset = matches.maxByOrNull { it.value.size }?.key ?: return null

        // Count audio_id occurrences at best offset
        val counter = matches[maxOffset]!!.groupingBy { it }.eachCount()
        val total = counter.values.sum()

        // Return sorted by confidence
        return counter.map { (audioId, count) ->
            MatchResult(audioId, count.toFloat() / total)
        }.sortedByDescending { it.confidence }
    }

    /**
     * Get metadata for an audio ID.
     * 
     * @param audioId Audio identifier
     * @return Metadata map or null
     */
    fun getMetadata(audioId: Int): Map<String, String>? = metadata[audioId]

    /**
     * Clear all fingerprints and metadata.
     */
    fun clear() {
        fingerprints.clear()
        metadata.clear()
    }
}
