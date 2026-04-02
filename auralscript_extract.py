"""
AuralScript Extractor v2.1
Converts an MP3/WAV file into a structured text schema that can be
read and analyzed by an LLM to identify genre, style, and sonic character.

Usage:
    python auralscript_extract.py "your_song.mp3"
    python auralscript_extract.py "your_song.mp3" --out "output.txt"
    python auralscript_extract.py "your_song.mp3" --processed   (DAW-processed files)

Requirements:
    pip install librosa numpy scipy soundfile

Optional (PANNs neural audio tagging):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install panns-inference
    Then download to C:\\Users\\<you>\\panns_data\\ :
      - class_labels_indices.csv  (from Google AudioSet)
      - Cnn14_mAP=0.431.pth      (from Zenodo record 3987831)

What AuralScript captures:
    META          BPM, key, tempo stability, beat strength
    ENERGY        RMS, dynamic range, loudness level
    BASS          Sub vs mid-bass balance, bass character
    ONSET DENSITY Arrangement busyness, hits per second
    SPECTRUM      Brightness, centroid, bandwidth
    DISTORTION    HF noise floor distortion estimate
    STRUCTURE     Quiet/loud pattern, verse-chorus detection
    HARMONIC      Chord complexity, tonal stability
    REVERB/ROOM   Decay time, electronic vs organic hint
    STEREO WIDTH  Mono vs wide mix
    TEXTURE       Harmonic/percussive balance, roughness, MFCCs
    VOCAL         Vocal presence and character in mix
    TRANSIENTS    Attack sharpness
    WAVEFORM      ASCII amplitude plot
    TIMELINE      16-segment breakdown with key, energy, vocal, pitch confidence
    PANNS TAGS    Neural audio tagging (optional, requires PANNs install)
"""

import librosa
import numpy as np
import argparse
import os
import sys
from datetime import datetime

PANNS_AVAILABLE = False  # default

# Optional PANNs integration
try:
    import panns_inference
    from panns_inference import AudioTagging, labels as panns_labels
    PANNS_AVAILABLE = True
except ImportError:
    pass

# Optional Whisper integration — install with:
#   pip install faster-whisper
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


def estimate_stereo_width(filepath):
    """Load stereo file and compute mid/side width. Returns None if mono."""
    try:
        y_stereo, sr = librosa.load(filepath, sr=None, mono=False)
        if y_stereo.ndim == 1:
            return None, None
        left = y_stereo[0]
        right = y_stereo[1]
        mid = (left + right) / 2
        side = (left - right) / 2
        mid_rms = float(np.sqrt(np.mean(mid**2)))
        side_rms = float(np.sqrt(np.mean(side**2)))
        if mid_rms < 1e-9:
            return 0.0, "Mono or near-mono"
        width_ratio = side_rms / mid_rms
        if width_ratio < 0.1:
            label = "Narrow/Mono (raw, punchy, old-school)"
        elif width_ratio < 0.3:
            label = "Slightly wide (centered mix)"
        elif width_ratio < 0.6:
            label = "Moderate stereo width (produced feel)"
        else:
            label = "Wide/Expansive (heavily produced or reverb-heavy)"
        return round(width_ratio, 3), label
    except Exception:
        return None, None


def extract(filepath, out_path=None, processed=False):
    print(f"[AuralScript] Loading: {filepath}")

    # Stereo width — must load before mono conversion
    stereo_width_ratio, stereo_label = estimate_stereo_width(filepath)

    # Load mono for all other analysis
    try:
        y, sr = librosa.load(filepath, sr=None, mono=True)
    except Exception as e:
        print(f"ERROR: Could not load file. {e}")
        sys.exit(1)

    duration = librosa.get_duration(y=y, sr=sr)
    print(f"[AuralScript] Duration: {duration:.1f}s  Sample rate: {sr}Hz")

    # ── TEMPO & BEAT ──────────────────────────────────────────────
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_strength = float(np.mean(librosa.onset.onset_strength(y=y, sr=sr)))

    # Tempo stability — measure BPM variance across 4 quarters
    quarter = len(y) // 4
    quarter_tempos = []
    for i in range(4):
        seg = y[i*quarter:(i+1)*quarter]
        t, _ = librosa.beat.beat_track(y=seg, sr=sr)
        quarter_tempos.append(float(np.atleast_1d(t)[0]))
    tempo_std = float(np.std(quarter_tempos))
    if tempo_std < 3:
        tempo_stability = f"Stable (+-{tempo_std:.1f} BPM variance)"
    elif tempo_std < 8:
        tempo_stability = f"Slight drift (+-{tempo_std:.1f} BPM variance)"
    else:
        tempo_stability = f"Unstable/wandering (+-{tempo_std:.1f} BPM variance)"

    # ── KEY & SCALE ───────────────────────────────────────────────
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                 'F#', 'G', 'G#', 'A', 'A#', 'B']
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    def detect_key(audio):
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        key_idx = int(np.argmax(chroma_mean))
        roll = np.roll(chroma_mean, -key_idx)
        maj_corr = np.corrcoef(roll, major_profile)[0, 1]
        min_corr = np.corrcoef(roll, minor_profile)[0, 1]
        scale = "Major" if maj_corr > min_corr else "Minor"
        return key_names[key_idx], scale

    key, scale = detect_key(y)

    # Keep chroma_mean in outer scope for harmonic complexity section
    chroma_full_global = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma_full_global, axis=1)

    # ── ENERGY ───────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_max = float(np.max(rms))
    rms_std = float(np.std(rms))

    # Fixed dynamic range — exclude near-silence frames (bottom 5%)
    rms_threshold = float(np.percentile(rms, 5))
    rms_active = rms[rms > rms_threshold]
    if len(rms_active) > 0:
        dynamic_range_db = float(20 * np.log10(
            (np.max(rms_active) + 1e-9) / (np.min(rms_active) + 1e-9)))
    else:
        dynamic_range_db = 0.0

    if rms_mean < 0.02:
        energy_label = "Very Low (ambient/sparse)"
    elif rms_mean < 0.05:
        energy_label = "Low (mellow/restrained)"
    elif rms_mean < 0.10:
        energy_label = "Medium (balanced)"
    elif rms_mean < 0.18:
        energy_label = "High (energetic/dense)"
    else:
        energy_label = "Very High (aggressive/saturated)"

    # ── BASS ENERGY ───────────────────────────────────────────────
    # Bandpass filter in time domain then measure RMS — stable and normalized
    from scipy.signal import butter, sosfilt

    def bandpass_rms(audio, low, high, sample_rate):
        """RMS energy of audio filtered to a frequency band."""
        nyq = sample_rate / 2
        low_n = max(low / nyq, 1e-4)
        high_n = min(high / nyq, 0.9999)
        if low_n >= high_n:
            return 0.0
        sos = butter(4, [low_n, high_n], btype='band', output='sos')
        filtered = sosfilt(sos, audio)
        return float(np.sqrt(np.mean(filtered ** 2)))

    sub_rms = bandpass_rms(y, 20, 80, sr)
    midbass_rms = bandpass_rms(y, 80, 250, sr)
    highmid_rms = bandpass_rms(y, 2000, 8000, sr)
    full_rms = float(np.sqrt(np.mean(y ** 2)))

    # Express as dB relative to full-signal RMS (avoids the STFT scale problem)
    def to_db_rel(band, reference):
        return float(20 * np.log10((band + 1e-9) / (reference + 1e-9)))

    sub_db_rel = to_db_rel(sub_rms, full_rms)
    midbass_db_rel = to_db_rel(midbass_rms, full_rms)
    highmid_db_rel = to_db_rel(highmid_rms, full_rms)
    bass_vs_ref = ((sub_db_rel + midbass_db_rel) / 2) - highmid_db_rel

    if bass_vs_ref < -6:
        bass_label = "Thin/bass-light (may lack punch)"
    elif bass_vs_ref < 0:
        bass_label = "Balanced bass presence"
    elif bass_vs_ref < 6:
        bass_label = "Bass-forward (warm, heavy low end)"
    else:
        bass_label = "Very bass-heavy (dominant low end)"

    sub_to_mid_db = sub_db_rel - midbass_db_rel
    if sub_to_mid_db > 2:
        bass_character = "Sub-dominant (felt more than heard — electronic/trap)"
    elif sub_to_mid_db < -4:
        bass_character = "Mid-bass dominant (guitar bass, punchy kick)"
    else:
        bass_character = "Balanced sub/mid-bass"

    sub_energy = sub_db_rel
    midbass_energy = midbass_db_rel
    bass_ratio = bass_vs_ref

    # ── ONSET DENSITY ────────────────────────────────────────────
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onset_density = len(onset_times) / duration

    if onset_density < 1.0:
        density_label = "Sparse (slow, open arrangement)"
    elif onset_density < 2.5:
        density_label = "Moderate (mid-density arrangement)"
    elif onset_density < 4.5:
        density_label = "Busy (active, layered arrangement)"
    else:
        density_label = "Very dense (rapid-fire hits, chaotic energy)"

    # ── SAMPLE / LOOP DETECTION ───────────────────────────────────

    # 1. Beat-tracked IOI variance — use beat frames not all onsets
    # Beat frames are the rhythmic grid, much more stable signal for quantization
    if len(beat_frames) > 4:
        beat_times_sec = librosa.frames_to_time(beat_frames, sr=sr)
        beat_ioi = np.diff(beat_times_sec)
        # Filter out outliers (gaps > 3x median = likely missed beat)
        median_ioi = float(np.median(beat_ioi))
        beat_ioi_clean = beat_ioi[beat_ioi < median_ioi * 3]
        if len(beat_ioi_clean) > 2:
            ioi_cv = float(np.std(beat_ioi_clean) / (np.mean(beat_ioi_clean) + 1e-9))
        else:
            ioi_cv = 0.0
    else:
        ioi_cv = 0.5  # not enough beats to measure

    if ioi_cv < 0.04:
        quantization_label = "Highly quantized (drum machine, programmed, or sampled loops)"
    elif ioi_cv < 0.08:
        quantization_label = "Moderately quantized (sequenced with slight human feel)"
    elif ioi_cv < 0.14:
        quantization_label = "Natural feel (live performance or humanized programming)"
    else:
        quantization_label = "Highly irregular (free tempo, rubato, or unstable beat)"

    # 2. Recurrence matrix using MFCCs — timbre repeats, not just harmony
    hop_rec = 4096
    mfcc_rec = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_rec)
    mfcc_norm = librosa.util.normalize(mfcc_rec, axis=0)
    rec_matrix = librosa.segment.recurrence_matrix(
        mfcc_norm, mode='affinity', sym=True,
        k=min(10, mfcc_norm.shape[1] - 1)
    )
    n = rec_matrix.shape[0]
    if n > 20:
        # Vectorized upper triangle excluding near-diagonal (k=10 ~ 3-4 seconds)
        upper = np.triu(rec_matrix, k=10)
        nonzero = upper[upper > 0]
        recurrence_score = float(np.mean(nonzero)) if len(nonzero) > 0 else 0.0
    else:
        recurrence_score = 0.0

    if recurrence_score > 0.25:
        recurrence_label = "Very high repetition (heavy looping — sample-based likely)"
    elif recurrence_score > 0.15:
        recurrence_label = "High repetition (loop-based or structured verse/chorus)"
    elif recurrence_score > 0.08:
        recurrence_label = "Moderate repetition (standard song structure)"
    else:
        recurrence_label = "Low repetition (through-composed, improvised, or collage)"

    # ── SPECTRUM ─────────────────────────────────────────────────
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    if spectral_centroid < 1500:
        brightness = "Dark/Warm (bass-heavy, muddy, or muffled)"
    elif spectral_centroid < 3000:
        brightness = "Balanced/Neutral"
    elif spectral_centroid < 5000:
        brightness = "Bright/Present (upper-mid forward)"
    else:
        brightness = "Very Bright/Airy (high shimmer, harsh potential)"

    # ── SPECTRAL FLATNESS (kept as data point, not used for distortion est) ──
    spec_flatness = librosa.feature.spectral_flatness(y=y)[0]
    flatness_mean = float(np.mean(spec_flatness))
    flatness_std = float(np.std(spec_flatness))
    # Note: flatness collapses on compressed/limited audio — use HF noise floor instead

    # ── DISTORTION (high-frequency noise floor approach) ──────────
    # Distorted guitars generate broadband noise/harmonics above 8kHz
    # This survives compression better than spectral flatness
    hf_rms = bandpass_rms(y, 8000, min(16000, sr // 2 - 100), sr)
    mid_rms_dist = bandpass_rms(y, 1000, 4000, sr)
    hf_to_mid_ratio = to_db_rel(hf_rms, mid_rms_dist + 1e-9)

    # Self-calibrate thresholds based on processed flag
    if processed:
        # DAW-processed files run hotter, compress HF — tighten thresholds
        dist_high = -8
        dist_mod = -14
        dist_low = -20
    else:
        dist_high = -6
        dist_mod = -12
        dist_low = -18

    if hf_to_mid_ratio > dist_high:
        distortion_est = "High — heavy distortion, fuzz, or feedback likely"
    elif hf_to_mid_ratio > dist_mod:
        distortion_est = "Moderate — overdrive or distorted power chords likely"
    elif hf_to_mid_ratio > dist_low:
        distortion_est = "Low — light crunch or clean with grit"
    else:
        distortion_est = "Minimal — likely clean, acoustic, or heavily filtered"

    # ── STRUCTURE DETECTION (self-calibrating) ────────────────────
    seg_rms_values = []
    struct_seg_count = 16
    struct_seg_len = len(y) // struct_seg_count
    for i in range(struct_seg_count):
        seg = y[i*struct_seg_len:(i+1)*struct_seg_len]
        seg_rms_values.append(float(np.mean(librosa.feature.rms(y=seg)[0])))

    rms_arr = np.array(seg_rms_values)
    rms_median = float(np.median(rms_arr))
    rms_sigma = float(np.std(rms_arr))

    # Use tighter sigma for processed files (compressed dynamic range)
    sigma_mult = 0.5 if processed else 0.75
    quiet_threshold = rms_median - (rms_sigma * sigma_mult)
    loud_threshold = rms_median + (rms_sigma * sigma_mult)

    pattern = []
    for v in rms_arr:
        if v < quiet_threshold:
            pattern.append(0)  # quiet
        elif v > loud_threshold:
            pattern.append(2)  # loud
        else:
            pattern.append(1)  # mid

    # Count transitions between quiet(0) and loud(2) ignoring mid
    extremes = [p for p in pattern if p != 1]
    transitions = sum(1 for i in range(1, len(extremes)) if extremes[i] != extremes[i-1])
    quiet_segs = pattern.count(0)
    loud_segs = pattern.count(2)
    mid_segs = pattern.count(1)

    if transitions >= 3 and quiet_segs >= 2 and loud_segs >= 2:
        structure_label = "Quiet/loud alternating (verse-chorus dynamic contrast)"
    elif transitions >= 2 and quiet_segs >= 1:
        structure_label = "Partial dynamic contrast (some quiet/loud movement)"
    elif rms_arr[-4:].mean() > rms_arr[:4].mean() + rms_sigma * 0.5:
        structure_label = "Linear build (gets progressively louder)"
    elif rms_arr[:4].mean() > rms_arr[-4:].mean() + rms_sigma * 0.5:
        structure_label = "Fades out (louder at start)"
    else:
        structure_label = "Consistent energy (minimal dynamic structure)"

    # ── TEXTURE / TIMBRE ─────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1).tolist()

    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy_val = float(np.mean(librosa.feature.rms(y=y_harm)[0]))
    perc_energy_val = float(np.mean(librosa.feature.rms(y=y_perc)[0]))
    if harm_energy_val > perc_energy_val * 1.5:
        texture_balance = "Harmonic-dominant (melodic/tonal content leads)"
    elif perc_energy_val > harm_energy_val * 1.5:
        texture_balance = "Percussive-dominant (rhythm/drums lead)"
    else:
        texture_balance = "Balanced harmonic/percussive mix"

    if zcr > 0.15:
        roughness = "High (distorted, noisy, or busy texture)"
    elif zcr > 0.08:
        roughness = "Medium (some grit or complexity)"
    else:
        roughness = "Low (smooth, clean, or sustained tones)"

    # ── HARMONIC COMPLEXITY ───────────────────────────────────────
    chroma_full = chroma_full_global  # reuse already computed chroma
    chroma_var = float(np.mean(np.var(chroma_full, axis=1)))

    # Entropy of mean chroma — but this picks up timbre not just harmony
    # so we weight chord change rate more heavily for the label
    chroma_entropy = float(-np.sum(
        np.where(chroma_mean > 0, chroma_mean / (chroma_mean.sum() + 1e-9) *
                 np.log2(chroma_mean / (chroma_mean.sum() + 1e-9) + 1e-9), 0)
    ))

    # Chord change rate — dominant pitch class shift per frame
    # This is more reliable than entropy for harmonic complexity
    chroma_frames = chroma_full
    dominant_per_frame = np.argmax(chroma_frames, axis=0)
    chord_changes = np.sum(np.diff(dominant_per_frame) != 0)
    chord_change_rate = chord_changes / (chroma_frames.shape[1] + 1e-9)

    # Label based primarily on chord change rate, entropy as secondary signal
    # Entropy alone is unreliable on synth-heavy tracks (overtones inflate it)
    if chord_change_rate < 0.12:
        harmonic_label = "Simple/repetitive (few chords, loops — electronic/pop/punk)"
    elif chord_change_rate < 0.20:
        if chroma_entropy < 3.1:
            harmonic_label = "Moderate complexity (standard verse-chorus harmony)"
        else:
            harmonic_label = "Moderate-complex (active harmony or rich timbre)"
    elif chord_change_rate < 0.30:
        harmonic_label = "Complex (many chord changes — rock, indie, jazz-adjacent)"
    else:
        harmonic_label = "Highly complex (dense harmony — jazz, prog, or modal)"

    # Tonal stability — does one key dominate or does it shift constantly
    top_chroma = float(np.max(chroma_mean) / (chroma_mean.sum() + 1e-9))
    if top_chroma > 0.15:
        tonal_stability = "Strong tonal center (key is clear and stable)"
    elif top_chroma > 0.10:
        tonal_stability = "Moderate tonal center (some ambiguity)"
    else:
        tonal_stability = "Weak tonal center (ambiguous or constantly shifting key)"

    # ── REVERB / ROOM SIZE ESTIMATE ───────────────────────────────
    # Use -10dB decay window instead of -20dB — more robust on compressed audio
    # -20dB was hitting the limiter floor instantly on brick-wall masters
    hop = 512
    rms_frames = librosa.feature.rms(y=y, hop_length=hop)[0]
    onset_frames_rev = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, units='frames')

    decay_times = []
    for of in onset_frames_rev[:30]:
        peak_idx = min(of + 2, len(rms_frames) - 1)
        peak_val = rms_frames[peak_idx]
        if peak_val < 1e-4:
            continue  # skip near-silent onsets
        target = peak_val * 10 ** (-10/20)  # -10dB (was -20dB — too aggressive for limiters)
        for j in range(peak_idx + 1, min(peak_idx + 200, len(rms_frames))):
            if rms_frames[j] <= target:
                decay_frames = j - peak_idx
                decay_time = decay_frames * hop / sr
                if decay_time > 0.001:  # ignore sub-millisecond readings
                    decay_times.append(decay_time)
                break

    mean_decay = float(np.mean(decay_times)) if decay_times else 0.05

    if mean_decay < 0.05:
        reverb_label = "Very dry (close-mic, anechoic — punk, lo-fi, tight electronic)"
    elif mean_decay < 0.15:
        reverb_label = "Dry/intimate (small room or tight production)"
    elif mean_decay < 0.30:
        reverb_label = "Moderate reverb (studio room or light hall)"
    elif mean_decay < 0.55:
        reverb_label = "Spacious (large room, ambient, or heavy reverb)"
    else:
        reverb_label = "Very wet/washy (shoegaze, ambient, cathedral — long tail)"

    # Electronic vs organic: synths have very consistent decay; organic varies
    if decay_times and len(decay_times) > 2:
        decay_std = float(np.std(decay_times))
        decay_consistency = decay_std / (mean_decay + 1e-9)
    else:
        decay_consistency = 0.0

    if decay_consistency < 0.3:
        instrument_hint = "Consistent decay (suggests synthesized/electronic sources)"
    elif decay_consistency < 0.6:
        instrument_hint = "Variable decay (mix of electronic and organic likely)"
    else:
        instrument_hint = "Highly variable decay (organic/live instruments likely)"
    vocal_rms = bandpass_rms(y, 300, 3000, sr)
    presence_rms = bandpass_rms(y, 3000, 8000, sr)
    vocal_db_rel = to_db_rel(vocal_rms, full_rms)
    presence_db_rel = to_db_rel(presence_rms, full_rms)
    vocal_vs_presence = vocal_db_rel - presence_db_rel

    if vocal_db_rel < -15:
        vocal_label = "Vocals likely buried or absent"
    elif vocal_db_rel < -8:
        vocal_label = "Vocals present but recessed"
    elif vocal_db_rel < -3:
        vocal_label = "Vocals well forward in mix"
    else:
        vocal_label = "Vocal range dominant (very forward or heavy midrange)"

    if vocal_vs_presence > 6:
        vocal_character = "Warm/boxy (lower mid dominant — may lack air)"
    elif vocal_vs_presence > 0:
        vocal_character = "Balanced vocal presence"
    else:
        vocal_character = "Bright/airy vocal range (presence forward)"

    # ── TRANSIENT SHARPNESS ───────────────────────────────────────
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # Peak-to-mean ratio of onset envelope — high = sharp attacks
    transient_ratio = float(np.max(onset_env) / (np.mean(onset_env) + 1e-9))
    # Attack time proxy — how quickly onsets rise
    onset_frames_all = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
    if len(onset_frames_all) > 1:
        # Mean rise time: frames from local min before onset to onset peak
        rise_times = []
        for of in onset_frames_all[:50]:  # sample first 50 onsets
            start = max(0, of - 5)
            window = onset_env[start:of+1]
            if len(window) > 1:
                rise_times.append(len(window))
        mean_rise = float(np.mean(rise_times)) if rise_times else 3.0
    else:
        mean_rise = 3.0

    if transient_ratio > 8 and mean_rise < 3:
        transient_label = "Very sharp/snappy (tight punk/metal attack)"
    elif transient_ratio > 5:
        transient_label = "Sharp attacks (energetic, punchy)"
    elif transient_ratio > 3:
        transient_label = "Moderate attack (balanced transients)"
    else:
        transient_label = "Soft/rounded attacks (ambient, compressed, or washy)"

    # ── TIMELINE (16 segments) ────────────────────────────────────
    segment_count = 16
    segment_len = len(y) // segment_count
    timeline = []
    for i in range(segment_count):
        start_sample = i * segment_len
        end_sample = start_sample + segment_len
        seg = y[start_sample:end_sample]
        seg_rms = float(np.mean(librosa.feature.rms(y=seg)[0]))
        seg_centroid = float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)))
        seg_time = start_sample / sr

        try:
            seg_key, seg_scale = detect_key(seg)
            root_idx = key_names.index(key)
            dominant_idx = (root_idx + 7) % 12
            dominant_name = key_names[dominant_idx]
            if seg_key == key:
                seg_key_str = f"{seg_key} {seg_scale}"
            elif seg_key == dominant_name:
                seg_key_str = f"{seg_key} {seg_scale} (V)"
            else:
                seg_key_str = f"{seg_key} {seg_scale} (!)"
        except Exception:
            seg_key_str = "?"

        seg_onsets = librosa.onset.onset_detect(y=seg, sr=sr)
        seg_dur = len(seg) / sr
        seg_density = len(seg_onsets) / seg_dur if seg_dur > 0 else 0

        # Per-segment vocal energy
        seg_vocal_rms = bandpass_rms(seg, 300, 3000, sr)
        seg_full_rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-9
        seg_vocal_db = to_db_rel(seg_vocal_rms, seg_full_rms)

        # Per-segment pitch confidence via pyin
        try:
            f0, voiced_flag, voiced_prob = librosa.pyin(
                seg,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            voiced_probs = voiced_prob[voiced_flag] if np.any(voiced_flag) else np.array([0.0])
            pitch_confidence = float(np.mean(voiced_probs)) if len(voiced_probs) > 0 else 0.0
        except Exception:
            pitch_confidence = 0.0

        # Key annotation with confidence-aware flagging
        try:
            seg_key, seg_scale = detect_key(seg)
            root_idx = key_names.index(key)
            dominant_idx = (root_idx + 7) % 12
            dominant_name = key_names[dominant_idx]
            if seg_key == key:
                seg_key_str = f"{seg_key} {seg_scale}"
                key_flag = ""
            elif seg_key == dominant_name:
                seg_key_str = f"{seg_key} {seg_scale} (V)"
                key_flag = ""
            else:
                # Unexpected key — check pitch confidence
                if pitch_confidence < 0.4:
                    seg_key_str = f"{seg_key} {seg_scale} (~)"  # low confidence = likely artifact
                    key_flag = "pitch artifact"
                else:
                    seg_key_str = f"{seg_key} {seg_scale} (!)"  # high confidence = real shift
                    key_flag = "key shift"
        except Exception:
            seg_key_str = "?"
            key_flag = ""
            pitch_confidence = 0.0

        timeline.append({
            "time": f"{int(seg_time//60):02d}:{seg_time%60:04.1f}",
            "rms": round(seg_rms, 4),
            "centroid_hz": round(seg_centroid, 1),
            "key": seg_key_str,
            "key_flag": key_flag,
            "pitch_conf": round(pitch_confidence, 2),
            "density": round(seg_density, 1),
            "vocal_db": round(seg_vocal_db, 1)
        })

    # ── ASCII WAVEFORM ────────────────────────────────────────────
    waveform_width = 60  # characters wide
    waveform_height = 8  # rows tall
    # Downsample RMS to waveform_width points
    rms_full = librosa.feature.rms(y=y, hop_length=512)[0]
    indices = np.linspace(0, len(rms_full) - 1, waveform_width).astype(int)
    rms_sampled = rms_full[indices]
    rms_norm = rms_sampled / (np.max(rms_sampled) + 1e-9)

    # Build grid top-down
    waveform_lines = []
    chars = ['█', '▇', '▆', '▅', '▄', '▃', '▂', '▁']
    for row in range(waveform_height, 0, -1):
        threshold = row / waveform_height
        line = ""
        for val in rms_norm:
            if val >= threshold:
                line += "█"
            elif val >= threshold - (1 / waveform_height):
                line += "▄"
            else:
                line += " "
        waveform_lines.append(f"  |{line}|")

    # Time axis label
    time_axis = "  0" + " " * (waveform_width - 4) + f"{int(duration//60)}:{int(duration%60):02d}"

    # ── PANNS AUDIO TAGGING (optional) ───────────────────────────
    panns_results = []
    panns_top_music = []
    panns_error = None

    if PANNS_AVAILABLE:
        try:
            print("[AuralScript] Running PANNs audio tagging...")
            y_panns = librosa.resample(y, orig_sr=sr, target_sr=32000)
            audio_panns = y_panns[None, :]

            at = AudioTagging(checkpoint_path=None, device='cpu')
            (clipwise_output, _) = at.inference(audio_panns)

            scores = clipwise_output[0]
            top_indices = np.argsort(scores)[::-1][:15]
            for idx in top_indices:
                if scores[idx] > 0.05:
                    panns_results.append((panns_labels[idx], float(scores[idx])))

            music_keywords = [
                'music', 'rock', 'pop', 'jazz', 'blues', 'metal', 'punk',
                'electronic', 'hip', 'folk', 'country', 'classical', 'dance',
                'guitar', 'drum', 'bass', 'piano', 'synth', 'vocal', 'sing',
                'beat', 'rhythm', 'melody', 'chord', 'grunge', 'indie',
                'ambient', 'house', 'techno', 'funk', 'soul', 'reggae',
                'distort', 'electric', 'acoustic', 'string', 'wind', 'brass',
                'yell', 'shout', 'angry', 'whoop', 'bellow', 'speech', 'rap'
            ]
            for label, score in panns_results:
                if any(kw in label.lower() for kw in music_keywords):
                    panns_top_music.append((label, score))

        except Exception as e:
            panns_error = str(e)
            print(f"[AuralScript] PANNs error: {e}")

    # ── WHISPER LANGUAGE & TRANSCRIPTION (optional) ───────────────
    whisper_language = None
    whisper_language_prob = None
    whisper_transcript = None
    whisper_error = None

    if WHISPER_AVAILABLE:
        try:
            print("[AuralScript] Running Whisper language detection...")
            model = WhisperModel("small", device="cpu", compute_type="int8")
            segments, info = model.transcribe(filepath, beam_size=5,
                                              language=None,  # auto-detect
                                              task="transcribe")
            whisper_language = info.language
            whisper_language_prob = round(info.language_probability, 3)

            # Collect transcript — up to 4000 chars
            transcript_parts = []
            char_count = 0
            for seg in segments:
                text = seg.text.strip()
                if not text:
                    continue
                transcript_parts.append(text)
                char_count += len(text)
                if char_count >= 4000:
                    break
            whisper_transcript = " ".join(transcript_parts)

        except Exception as e:
            whisper_error = str(e)
            print(f"[AuralScript] Whisper error: {e}")

    # ── ASSEMBLE AURALSCRIPT ──────────────────────────────────────
    def duration_str_short(d):
        return f"{int(d//60)}:{int(d%60):02d}"

    filename = os.path.basename(filepath)
    lines = []
    lines.append("=" * 65)
    lines.append("AURALSCRIPT v2.3")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Source:    {filename}")
    lines.append(f"Mode:      {'DAW-Processed (--processed)' if processed else 'Raw Suno Output'}")
    lines.append("=" * 65)

    lines.append("\n── META ──────────────────────────────────────────────────────")
    lines.append(f"  Duration:            {int(duration//60)}:{duration%60:04.1f}")
    lines.append(f"  BPM:                 {tempo:.1f}")
    lines.append(f"  Tempo Stability:     {tempo_stability}")
    lines.append(f"  Key:                 {key} {scale}")
    lines.append(f"  Sample Rate:         {sr} Hz")
    lines.append(f"  Beat Strength:       {beat_strength:.3f}")

    lines.append("\n── ENERGY ────────────────────────────────────────────────────")
    lines.append(f"  Level:               {energy_label}")
    lines.append(f"  RMS Mean:            {rms_mean:.4f}")
    lines.append(f"  RMS Peak:            {rms_max:.4f}")
    lines.append(f"  Dynamic Range:       {dynamic_range_db:.1f} dB  (active frames only)")
    lines.append(f"  Consistency:         {'Consistent (low variance)' if rms_std < 0.02 else 'Variable (high dynamics)'}")

    lines.append("\n── BASS ──────────────────────────────────────────────────────")
    lines.append(f"  Presence:            {bass_label}")
    lines.append(f"  Character:           {bass_character}")
    lines.append(f"  Bass vs Hi-Mid:      {bass_vs_ref:+.1f} dB  (0=balanced, +=bass-heavy, -=thin)")
    lines.append(f"  Sub (20-80Hz):       {sub_energy:+.1f} dB  rel to full signal")
    lines.append(f"  Mid-Bass (80-250Hz): {midbass_energy:+.1f} dB  rel to full signal")

    lines.append("\n── ONSET DENSITY ─────────────────────────────────────────────")
    lines.append(f"  Arrangement:         {density_label}")
    lines.append(f"  Onsets/sec:          {onset_density:.2f}")
    lines.append(f"  Total Onsets:        {len(onset_times)}")

    lines.append("\n── SAMPLE / LOOP DETECTION ───────────────────────────────────")
    lines.append(f"  Quantization:        {quantization_label}")
    lines.append(f"  IOI Variance (CV):   {ioi_cv:.3f}  (0=machine-perfect, >0.5=human/free)")
    lines.append(f"  Recurrence:          {recurrence_label}")
    lines.append(f"  Recurrence Score:    {recurrence_score:.3f}  (higher = more looping/repetition)")

    lines.append("\n── SPECTRUM ──────────────────────────────────────────────────")
    lines.append(f"  Brightness:          {brightness}")
    lines.append(f"  Centroid:            {spectral_centroid:.1f} Hz")
    lines.append(f"  Bandwidth:           {spectral_bandwidth:.1f} Hz")
    lines.append(f"  Rolloff (85%):       {spectral_rolloff:.1f} Hz")

    lines.append("\n── DISTORTION / FLATNESS ─────────────────────────────────────")
    lines.append(f"  Distortion Est:      {distortion_est}")
    lines.append(f"  HF/Mid Ratio:        {hf_to_mid_ratio:+.1f} dB  (8-16kHz vs 1-4kHz)")
    lines.append(f"  Flatness Mean:       {flatness_mean:.4f}  (data only — unreliable on compressed audio)")
    lines.append(f"  Flatness Variance:   {flatness_std:.4f}")

    lines.append("\n── STRUCTURE ─────────────────────────────────────────────────")
    lines.append(f"  Pattern:             {structure_label}")
    lines.append(f"  Loud Segments:       {loud_segs}/16")
    lines.append(f"  Mid Segments:        {mid_segs}/16")
    lines.append(f"  Quiet Segments:      {quiet_segs}/16")
    lines.append(f"  Transitions:         {transitions}")

    lines.append("\n── HARMONIC COMPLEXITY ───────────────────────────────────────")
    lines.append(f"  Complexity:          {harmonic_label}")
    lines.append(f"  Tonal Stability:     {tonal_stability}")
    lines.append(f"  Chroma Entropy:      {chroma_entropy:.3f}  (higher = more harmonic variety)")
    lines.append(f"  Chord Change Rate:   {chord_change_rate:.3f}  (higher = faster changes)")

    lines.append("\n── REVERB / ROOM ─────────────────────────────────────────────")
    lines.append(f"  Room Size:           {reverb_label}")
    lines.append(f"  Mean Decay Time:     {mean_decay:.3f}s  (to -20dB after transient)")
    lines.append(f"  Instrument Hint:     {instrument_hint}")
    lines.append(f"  Decay Consistency:   {decay_consistency:.3f}  (lower = more electronic)")

    lines.append("\n── STEREO WIDTH ──────────────────────────────────────────────")
    if stereo_label:
        lines.append(f"  Width:               {stereo_label}")
        lines.append(f"  Side/Mid Ratio:      {stereo_width_ratio}")
    else:
        lines.append(f"  Width:               Could not determine (mono source or load error)")

    lines.append("\n── TEXTURE ───────────────────────────────────────────────────")
    lines.append(f"  Balance:             {texture_balance}")
    lines.append(f"  Roughness:           {roughness}")
    lines.append(f"  ZCR:                 {zcr:.4f}")
    lines.append(f"  MFCC[1-5]:           {[round(x,2) for x in mfcc_means[:5]]}")
    lines.append(f"  MFCC[6-13]:          {[round(x,2) for x in mfcc_means[5:]]}")

    lines.append("\n── VOCAL PRESENCE ────────────────────────────────────────────")
    lines.append(f"  Level:               {vocal_label}")
    lines.append(f"  Character:           {vocal_character}")
    lines.append(f"  Vocal Range dB:      {vocal_db_rel:+.1f} dB  rel to full signal")
    lines.append(f"  Presence Range dB:   {presence_db_rel:+.1f} dB  rel to full signal")

    lines.append("\n── TRANSIENTS ────────────────────────────────────────────────")
    lines.append(f"  Attack:              {transient_label}")
    lines.append(f"  Peak/Mean Ratio:     {transient_ratio:.2f}  (higher = snappier)")
    lines.append(f"  Mean Rise Frames:    {mean_rise:.1f}")

    lines.append("\n── WAVEFORM (amplitude over time) ───────────────────────────")
    for wl in waveform_lines:
        lines.append(wl)
    lines.append(time_axis)

    lines.append("\n── TIMELINE (16 segments) ────────────────────────────────────")
    lines.append(f"  {'Time':<8} {'RMS':<8} {'Centroid':<12} {'Key':<16} {'Den/s':<7} {'VocaldB':<9} {'PConf':<7} Notes")
    lines.append(f"  {'-'*78}")
    prev_rms = None
    for seg in timeline:
        notes = []
        if prev_rms is not None:
            delta = seg['rms'] - prev_rms
            if delta > 0.02:
                notes.append("energy up")
            elif delta < -0.02:
                notes.append("energy down")
        if seg.get('key_flag'):
            notes.append(seg['key_flag'])
        note_str = ", ".join(notes)
        lines.append(
            f"  {seg['time']:<8} {seg['rms']:<8} {seg['centroid_hz']:<12} "
            f"{seg['key']:<16} {seg['density']:<7} {seg['vocal_db']:+.1f}      "
            f"{seg['pitch_conf']:.2f}   {note_str}"
        )
        prev_rms = seg['rms']

    lines.append("\n── PANNS AUDIO TAGS ──────────────────────────────────────────")
    if not PANNS_AVAILABLE:
        lines.append("  Status:              Not installed (pip install panns-inference torch)")
    elif panns_error:
        lines.append(f"  Status:              Error — {panns_error}")
    else:
        lines.append(f"  Status:              OK — {len(panns_results)} tags above 5% confidence")
        if panns_top_music:
            lines.append("  Music-relevant tags:")
            for label, score in panns_top_music[:10]:
                bar = "█" * int(score * 20)
                lines.append(f"    {score:.3f} {bar:<20} {label}")
        else:
            lines.append("  Music-relevant tags: None above threshold")
        if panns_results:
            lines.append("  All top tags:")
            for label, score in panns_results[:10]:
                lines.append(f"    {score:.3f}  {label}")

    lines.append("\n── WHISPER LANGUAGE & LYRICS ─────────────────────────────────")
    if not WHISPER_AVAILABLE:
        lines.append("  Status:              Not installed (pip install faster-whisper)")
    elif whisper_error:
        lines.append(f"  Status:              Error — {whisper_error}")
    else:
        lang_name = whisper_language.upper() if whisper_language else "Unknown"
        lines.append(f"  Language:            {lang_name}  (confidence: {whisper_language_prob})")
        lines.append(f"  Transcript ({len(whisper_transcript)} chars):")
        if whisper_transcript:
            # Word-wrap at 60 chars
            words = whisper_transcript.split()
            line_buf = "    "
            for word in words:
                if len(line_buf) + len(word) + 1 > 64:
                    lines.append(line_buf)
                    line_buf = "    " + word
                else:
                    line_buf += " " + word if line_buf.strip() else "    " + word
            if line_buf.strip():
                lines.append(line_buf)
        else:
            lines.append("    [no vocals detected]")

    lines.append("\n── SUNO TAGS (fill in after listening) ───────────────────────")
    lines.append("  Style Prompt:        [paste your Suno style prompt here]")
    lines.append("  Exclude Tags:        [paste your Suno exclude tags here]")
    lines.append("  Lyrics:              [paste lyrics here or 'see attached']")
    lines.append("  Suno Version:        [v3 / v4 / v5 / v5.5]")
    lines.append("  Subjective:          [how did it actually sound vs intent?]")
    lines.append("  Rating:              [1-10]")
    lines.append("  Notes:               [anything notable — drift, surprises, wins]")

    lines.append("\n" + "=" * 65)

    output = "\n".join(lines)

    if out_path is None:
        base = os.path.splitext(filepath)[0]
        out_path = base + "_auralscript.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"[AuralScript] Saved to: {out_path}")
    print("\n" + output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AuralScript Extractor v1.7")
    parser.add_argument("filepath", help="Path to MP3 or WAV file")
    parser.add_argument("--out", help="Output .txt path (optional)", default=None)
    parser.add_argument("--processed", action="store_true",
                        help="Flag if file has been through a DAW (adjusts thresholds)")
    args = parser.parse_args()
    extract(args.filepath, args.out, processed=args.processed)
