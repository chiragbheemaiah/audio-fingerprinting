import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
from collections import Counter, defaultdict

# --- Config ---
SHOW_DB = False  # True -> plot in dB (easier to see); False -> linear power (good for feature pipelines)
audio_id : int = 0
SHOW_VIZ = False

audio_db = dict()
hash_fingerprints = defaultdict(list) # (f2, f1, (t2-t1)) -> audio_id
# query_hash_fingerprints = defaultdict(list)
def reset():
    global audio_id
    auto_id = 0
    global audio_db 
    audio_db = dict()
    global hash_fingerprints
    hash_fingerprints = defaultdict(list)

def save_to_db(hashes, hash_fingerprints):
    for digest, audio_id, time_x in hashes:
        hash_fingerprints[digest].append((audio_id, time_x))

def read_audio_file(filepath: str):
    """
    Loads audio as mono float32 using librosa.
    Returns (sample_rate, audio_waveform)
    """
    if not os.path.exists(filepath):
        print(f"Error: The file {filepath} does not exist")
        return None, None
    try:
        # sr=None keeps original sample rate; mono=True folds to mono
        audio, sr = librosa.load(filepath, sr=None, mono=True)
        return sr, audio.astype(np.float32)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred while reading the file - {e}")
        return None, None

def viz_wave(audio):
    x_axis = np.arange(len(audio))
    y_axis = audio
    plt.figure(figsize=(10, 4))
    plt.plot(x_axis, y_axis)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title("Raw Audio Wave")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_mel_spectrogram(audio, sr, n_fft=2048, hop_length=512, n_mels=128, power=2.0):
    """
    Returns mel spectrogram (power if power=2.0, magnitude if power=1.0).
    Shape: (n_mels, time_frames)
    """
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,       # 2.0 -> power, 1.0 -> magnitude
    )
    return S

def viz_mel_spectrogram(S, sr, hop_length, show_db=False):
    """
    Visualize mel spectrogram in linear power or dB.
    """
    if show_db:
        # Convert power to dB for visualization convenience
        S_plot = librosa.power_to_db(S, ref=np.max)
        cbar_label = "Power (dB)"
        title = "Mel Spectrogram (dB)"
    else:
        # Linear power; optionally normalize for visibility
        S_plot = S / (S.max() + np.finfo(float).eps)
        cbar_label = "Normalized Power (Linear)"
        title = "Mel Spectrogram (Linear)"

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(
        S_plot,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel'
    )
    plt.colorbar(label=cbar_label)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def viz_constellation(data):
    x_axis = [i for i, _, _ in data]
    y_axis = [j for _, j, _ in data]
    plt.figure(figsize=(10, 8))
    plt.title("Constellation Diagram")
    plt.xlabel("Freq")
    plt.ylabel("Time")
    plt.scatter(x_axis, y_axis)
    plt.show()


def compute_constellations(spectrogram, WINDOW_SIZE):
    '''
    Calculate in a window of size WINDOW_SIZE, the max element and its position.
    '''
    max_points = []
    r, c = len(spectrogram), len(spectrogram[0])
    for i in range(0, r, WINDOW_SIZE):
        for j in range(0, c, WINDOW_SIZE):
            max_ele = -1
            max_k= -1
            max_l = -1
            for k in range(i, i + WINDOW_SIZE):
                for l in range(j, j + WINDOW_SIZE):
                    if k >= r or l >= c:
                        continue
                    if spectrogram[k][l] > max_ele:
                        max_ele = spectrogram[k][l]
                        max_k = k
                        max_l = l
            max_points.append([max_k, max_l, spectrogram[max_k][max_l]])
    return max_points

def filter_constellation(constellation, THRESH):
    sorted_constellation = sorted(constellation, key=lambda c : c[2])
    filter_idx = len(sorted_constellation) * THRESH
    return sorted_constellation[int(filter_idx):]

# --- Fuzzy hash helper (Shazam-style) ---
FUZ_F_HZ = 20.0    # frequency fuzz (≈ your FFT/mel bin spacing)
FUZ_DT_S = 0.02    # time fuzz (≈ your hop size in seconds)

def make_fuzzy_hash(f1_hz: float, f2_hz: float, dt_s: float) -> int:
    """
    Create a fuzzy composite hash from (f1, f2, Δt) similar to Shazam.

    1. Quantizes frequency and time difference values to coarse buckets.
    2. Packs them into one integer for fast, noise-tolerant lookup.
    """
    f1_q = f1_hz - (f1_hz % FUZ_F_HZ)
    f2_q = f2_hz - (f2_hz % FUZ_F_HZ)
    dt_q = dt_s  - (dt_s  % FUZ_DT_S)

    # Convert to small integer buckets
    b1 = int(f1_q // FUZ_F_HZ)
    b2 = int(f2_q // FUZ_F_HZ)
    b3 = int(dt_q // FUZ_DT_S)

    # Combine into one integer hash (p4 removed)
    return (b3 * 10**6) + (b2 * 10**3) + b1


def create_fingerprint(constellation, times, freq, TIME_WINDOW, audio_id, FUZZ_FACTOR = 2):
    hashes = []
    for x, (xi, xj, _) in enumerate(constellation):
        for y, (yi, yj, _) in enumerate(constellation):
            if x != y:
                time_x = times[xj]
                time_y = times[yj]
                freq_x = freq[xi]
                freq_y = freq[yi]
                time_diff = time_y - time_x
                if time_diff <= TIME_WINDOW:
                    digest = make_fuzzy_hash(freq_x, freq_y, time_diff)
                    hashes.append((digest, audio_id, time_x))
                    # hash_fingerprints[digest].append((audio_id, time_x))

    return hashes


        

def processing_pipeline(AUDIO_FILE_PATH: str):
    print("Reading in audio file...")
    sample_rate, audio = read_audio_file(AUDIO_FILE_PATH)
    if sample_rate is None or audio is None:
        return
    global audio_id
    audio_id += 1
    song_name = AUDIO_FILE_PATH.split('/')[-1]
    global audio_db
    audio_db[audio_id] = song_name
    print("Sample Rate:", sample_rate)
    print("Audio Data Shape:", audio.shape)


    # Optional: visualize waveform
    if SHOW_VIZ:
        print("Visualizing Audio....")
        viz_wave(audio)

    print("Computing mel spectrogram...")
    n_fft = 2048
    hop_length = 512
    n_mels = 128

    S_mel_db = compute_mel_spectrogram(
        audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,  # power mel-spectrogram
    )
    print("Mel spectrogram shape:", S_mel_db.shape)
    if SHOW_VIZ:
        print("Visualizing mel spectrogram...")
        viz_mel_spectrogram(S_mel_db, sr=sample_rate, hop_length=hop_length, show_db=SHOW_DB)

    S_mel = compute_mel_spectrogram(
        audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=1.0,  # linear mel-spectrogram
    )

    times = librosa.frames_to_time(np.arange(S_mel.shape[1]), sr=sample_rate, hop_length=hop_length)
    freq = librosa.mel_frequencies(S_mel.shape[0], fmin=0, fmax=sample_rate/2)

    print("Calculating constellation peaks......")
    WINDOW_SIZE = 15
    constellation = compute_constellations(S_mel, WINDOW_SIZE)

    if SHOW_VIZ:
        print("Visualizing constellation")
        viz_constellation(constellation)

    THRESHOLD = 0.2 # drop 20% of peaks
    filtered_constellations = filter_constellation(constellation, THRESHOLD)

    print("Length of filtered constellation: ", len(filtered_constellations))
    
    if SHOW_VIZ:
        print("Visualizing filtered constellation...........")
        viz_constellation(filtered_constellations)

    TIME_WINDOW = 5
    print("Commencing finderprinting process.....")
    hashes = create_fingerprint(filtered_constellations, times, freq, TIME_WINDOW, audio_id)
    print("Fingerprinting completed............")
    return hashes


    # return (filtered_constellations, times, freq, audio_id)


def evaluate(query_hashes, hash_fingerprints):
    # TODO: Using keys to compare is too sparse either design a semantic hash function or bin
    matches = defaultdict(list)
    for digest, _, time_x in query_hashes:
        if hash_fingerprints.get(digest, None) != None:
            # matching fingerprint
            for match in hash_fingerprints[digest]:
                anchor_audio_id = match[0]
                anchor_time_db = match[1]
                anchor_time_query = time_x
                target_time = anchor_time_db - anchor_time_query
                # CHECK: Quantizing to int for now
                matches[int(target_time)].append(anchor_audio_id)

    # Within the matches, choose the list with the most matches and predict the prob of the results.
    max_key = None
    max_len = 0
    for key in matches.keys():
        res = matches[key]
        if len(res) > max_len:
            max_len = len(res)
            max_key = key
    if max_len == 0 or max_key is None:
        return None

    res = []
    counter = Counter(matches[max_key])
    total_preds = sum(counter.values())

    for k, v in counter.items():
        prob = v / total_preds
        res.append((k, prob))

    return res
    

def predict():
    print("Commencing processing phase...................")

    AUDIO_DIR = "/home/blitz/Desktop/audio-fingerprinting/app/uploads"
    for filename in os.listdir(AUDIO_DIR):
        AUDIO_FILE_PATH = os.path.join(AUDIO_DIR, filename)
        print("Procssing audio file: ", AUDIO_FILE_PATH)
        hashes = processing_pipeline(AUDIO_FILE_PATH)
        save_to_db(hashes, hash_fingerprints)
        print("Saved Hashed Length: ", len(hash_fingerprints))
   
    print("-----------------------------------------------------------------------------")
   
    print("Prediction Pipeline Commencing.....")

    print("Enter query audio sample")
    QUERY_DIR = "/home/blitz/Desktop/audio-fingerprinting/app/query"
    for filename in os.listdir(QUERY_DIR):
        QUERY_SAMPLE = os.path.join(QUERY_DIR, filename)
        query_hashes = processing_pipeline(QUERY_SAMPLE)

        print("Starting evaluation......")
        results = evaluate(query_hashes, hash_fingerprints)
        response = []
        if results is None:
            print("No matches found :(")
            return
        
        for res in results:
            song_id = res[0]
            pred_prob = res[1]
            print(f"\nThe identified song is: {audio_db[song_id]} with song id: {song_id} with a probability of {pred_prob:.5f}")
            response.append({
                "audio_id" : song_id,
                "label": audio_db[song_id],
                "probability": pred_prob
            })


    reset()
    return response

def main():
    predict()
    
if __name__ == "__main__":
    main()
