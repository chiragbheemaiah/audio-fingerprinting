# Building My Own Audio Fingerprinting System

I have always wondered how Shazam can listen to a few seconds of a song and instantly tell you what it is. I decided to find out by building my own version, a lightweight and transparent audio fingerprinting system that actually explains itself instead of hiding behind mystery.

This post is a record of my journey designing, debugging, and understanding a system that can identify short audio clips based purely on their spectral fingerprints. It is part science experiment, part signal processing exploration, and part coding challenge.

---

## The Big Idea

The goal was simple on paper: take a short clip of sound, extract its unique fingerprint, and check if it matches something in a small library. Real-world audio, however, is messy. It can be noisy, start midway through, or be recorded at different volumes. I wanted something that could handle those imperfections.

Rather than using deep learning, I went with a clean and interpretable approach inspired by Shazam's original algorithm. The final system uses classical signal processing, mel-spectrograms, and some math that feels like detective work.

---

## Mel-Spectrograms: Seeing Sound

Audio signals are just sequences of numbers changing rapidly over time, which is not very intuitive. To visualize and analyze them, I converted them into mel-spectrograms, which show how energy varies across frequencies over time.

The mel scale represents sound in a way that matches how human hearing perceives pitch. It spaces frequencies unevenly, with finer resolution at low frequencies and coarser resolution at high ones. I used `n_fft = 2048`, `hop_length = 512`, and `n_mels = 128`. This setup provides enough detail without overloading the system.

When I plotted my first spectrogram, I realized it looked like an abstract painting that could sing.

---

## Finding the Constellations

After generating the spectrogram, the next task was to find the most distinctive points. I divided the spectrogram into small tiles and picked the highest-energy value in each. At first, it produced too many peaks and the result looked random. After tuning the parameters, I got a clear pattern of bright points scattered across the spectrogram. These peaks formed constellations that uniquely represented each sound.

This method worked because it captured both brief and continuous features of a sound while avoiding redundant information from louder sections.

---

## Turning Peaks into Fingerprints

The next challenge was to represent these constellations in a way that could be searched quickly. I paired each peak with others that appeared within a few seconds in time and encoded their relationship based on frequency and time difference.

For two peaks `(f1, f2)` separated by `Δt`, I generated an integer hash like this:

```
hash = (Δt_bucket * 10^6) + (f2_bucket * 10^3) + f1_bucket
```

By grouping values into buckets, I made the system tolerant to small frequency and timing variations. Even if a recording started later or included some noise, the relationships between peaks stayed consistent. This approach allowed the same sound to produce the same fingerprints under slightly different conditions.

---

## Matching the Query

When a new clip is processed, the system computes its hashes and compares them with those in the reference library. Each matching hash votes for a possible time offset between the query and the reference file. If many hashes agree on the same offset, it indicates a strong match.

This voting mechanism is simple but effective. One match can be a coincidence, but many matching offsets tell a consistent story.

---

## Keeping It Simple and Fast

I used FastAPI because it is lightweight, fast, and easy to work with for file uploads. Files are streamed in chunks to avoid memory issues, and I set a 50 MB limit to prevent oversized uploads. Everything else happens in memory, which keeps response times quick for small libraries.

No complex frameworks or databases were needed. The entire system runs on Python with NumPy and Librosa, and the logic is clear enough to follow line by line.

---

## Performance

For clips under thirty seconds, the processing feels instant. The most time-consuming step is computing the spectrogram, but it is still efficient on a standard CPU. The memory footprint is small because each fingerprint is stored as a compact integer. For small datasets, it performs perfectly.

It is not built for millions of tracks yet, but that was never the goal. This project is meant to be clear, inspectable, and educational.

---

## Improvements I Want to Make

There are plenty of directions to take this further. I could store fingerprints in a small database, use adaptive thresholds for peak selection, or apply weights to reduce false positives. Vectorizing the fingerprint generation with Numba or NumPy would make it faster, and adding automated tests would help keep it consistent.

Someday, I want to try running this on an ESP32. It would be fun to have a small device that hears a beep and says exactly which appliance made it.

---

## What I Learned

Building this system helped me understand sound on a deeper level. Mel-spectrograms reveal how frequencies evolve, while the constellation and hashing method show how structure emerges from noise. The time-offset voting step ties it all together in a surprisingly reliable way.

It is not magic. It is a clever mix of math and pattern recognition, and when the system correctly identifies a whistle as something it has heard before, it feels incredibly rewarding.
