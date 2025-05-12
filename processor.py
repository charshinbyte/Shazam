import os
import utils
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from collections import defaultdict
from mongodb_client import collection
from pymongo import UpdateOne 
import sqlite3

def generate_spectrogram(audio_path, n_fft, hop_length):
    y, sr = librosa.load(audio_path)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    return D, sr, n_fft

def spectrogram_band(D, sr, freq_range, n_fft):
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    min_freq, max_freq = freq_range
    freq_indices = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
    return D[freq_indices, :], freqs[freq_indices]

def plot_spectrogram(D, f, sr, hop_length):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='magma', y_coords=f)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()

def get_energy_peaks(D):
    peaks = np.zeros_like(D)
    max_indices = np.argmax(D, axis=0)
    for col, row in enumerate(max_indices):
        peaks[row, col] = D[row, col]
    
    nonzero_vals = peaks[peaks < 0]
    if len(nonzero_vals) > 0:
        threshold = np.percentile(nonzero_vals, 80)
        peaks = np.where(peaks >= threshold, peaks, 0)

    peaks = abs(peaks)
    neighborhood_size = (10, 10)
    min_value_threshold = 1e-6

    filtered_max = maximum_filter(peaks, size=neighborhood_size, mode='constant')
    local_max = (peaks == filtered_max)
    nonzero_mask = (peaks > min_value_threshold)

    return local_max & nonzero_mask

def get_fingerprints(D, n_fft, sr, bands, hop_length):
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    freq_indices = np.where((freqs >= bands[0]) & (freqs <= bands[-1]))[0]
    freqs_band = freqs[freq_indices]

    peaks = []
    for i in range(len(bands) - 1):
        D1, _ = spectrogram_band(D, sr, (bands[i], bands[i + 1]), n_fft)
        peak_band = get_energy_peaks(D1)
        peaks.append(peak_band)

    fingerprints = np.vstack(peaks)
    i, j = np.where(fingerprints != 0)
    duration = D.shape[1] * hop_length / sr
    time = np.linspace(0, duration, D.shape[1])

    fingerprint_times = time[j]
    fingerprint_freqs = freqs_band[i]

    sorted_indices = np.argsort(fingerprint_times)
    return fingerprint_times[sorted_indices], fingerprint_freqs[sorted_indices]

def generate_hash(times, freqs, song_id, fan_value, max_time_delta):
    hash_map = {}
    num_peaks = len(times)

    for i in range(num_peaks):
        anchor_time, anchor_freq = times[i], freqs[i]
        count, j = fan_value, 1

        while count > 0 and i + j < num_peaks:
            target_time, target_freq = times[i + j], freqs[i + j]
            delta_time = target_time - anchor_time
            if delta_time > max_time_delta:
                break
            if target_time != anchor_time:
                hash_key = (int(anchor_freq), int(target_freq), round(delta_time, 1))
                metadata = {"anchor_time": anchor_time, "song_id" : song_id}
                hash_map.setdefault(hash_key, []).append(metadata)
                count -= 1
            j += 1
    return hash_map

def song2hash(audio, song_id, config):
    D, sr, n_fft = generate_spectrogram(audio, config.n_fft, config.hop_length)
    times, freqs = get_fingerprints(D, n_fft, sr, config.bands, config.hop_length)
    return generate_hash(times, freqs, song_id, config.fan_value, config.max_time_delta)

def clip2hash(audio, config):
    D, sr, n_fft = generate_spectrogram(audio, config.n_fft, config.hop_length)
    times, freqs = get_fingerprints(D, n_fft, sr, config.bands, config.hop_length)

    hash_map = {}
    num_peaks = len(times)

    for i in range(num_peaks):
        anchor_time, anchor_freq = times[i], freqs[i]
        count, j = config.fan_value, 1

        while count > 0 and i + j < num_peaks:
            target_time, target_freq = times[i + j], freqs[i + j]
            delta_time = target_time - anchor_time
            if delta_time > config.max_time_delta:
                break
            if target_time != anchor_time:
                hash_key = (int(anchor_freq), int(target_freq), round(delta_time, 1))
                hash_map.setdefault(hash_key, []).append({"anchor_time": anchor_time})
                count -= 1
            j += 1
    return hash_map

async def find_matches(clipHash):
    offset_dict = defaultdict(lambda: defaultdict(int))
    
    keys = [str(key) for key in clipHash]
    documents = await collection.find({"_id": {"$in": keys}}).to_list(length=None)
    doc_map = {doc["_id"]: doc for doc in documents}

    for count, (key, metadata_list) in enumerate(clipHash.items(), 1):
        document = doc_map.get(str(key))
        if document:
            for song in document["metadata"]:
                song_id = song['song_id']
                song_time = song['anchor_time']

                for clip in metadata_list:
                    clip_time = clip['anchor_time']
                    offset = round(song_time - clip_time, 1)
                    offset_dict[song_id][offset] += 1
        else:
            print(f"No Key in Library {key}")
        
        if count % 100 == 0:
            print(f'{count} fingerprints processed')

    return {
        song_id: max(offset_counts.values())
        for song_id, offset_counts in offset_dict.items()
    }

async def match_clip(audio_path, config):
    fingerprints = clip2hash(audio=audio_path, config=config)
    print(f"Total Number of fingerprints {len(fingerprints)}")
    match_scores = await find_matches(fingerprints)
    sorted_scores = sorted(match_scores.items(), key=lambda x: x[1], reverse=True)

    if sorted_scores[0][1] < 30:
        return "Not able to find a match"
    else:
        song_id =  sorted_scores[0][0]
        conn = sqlite3.connect("data/songs.db")
        song_info = id_to_song(song_id, conn)
        if song_info:
            return (f"Song: {song_info[0]}, Artist: {song_info[1]}")
    

async def bulk_save_hash(hashed_song):
    operations = []

    for hash_key, metadata_list in hashed_song.items():
        doc_id = str(hash_key)
        operations.append(
            UpdateOne(
                {"_id": doc_id},
                {"$push": {"metadata": {"$each": metadata_list}}},
                upsert=True
            )
        )

    if operations:
        result = await collection.bulk_write(operations, ordered=False)
        return result 
    
async def upload_song(audio, song_id, config):
    utils.convert_mp3_to_wav(audio, "Upload.wav")

    hashed_song = song2hash("Upload.wav", song_id, config)
    await bulk_save_hash(hashed_song)

async def delete_songs():
    collection.delete_many({})

def save_song_to_sqlite(db_path: str, song_id: str, artist: str, song_name: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS songs (
            song_id TEXT PRIMARY KEY,
            artist TEXT,
            song_name TEXT
        )
    ''')

    cursor.execute('''
        INSERT OR REPLACE INTO songs (song_id, artist, song_name)
        VALUES (?, ?, ?)
    ''', (song_id, artist, song_name))

    conn.commit()
    conn.close()

def id_to_song(song_id, db):
    try:
        cursor = db.cursor() if hasattr(db, 'cursor') else db
        cursor.execute("SELECT song_name, artist FROM songs WHERE song_id = ?", (song_id,))
        result = cursor.fetchone()
        return result
    
    except Exception as e:
        print(f"Error retrieving song: {e}")
        return None
