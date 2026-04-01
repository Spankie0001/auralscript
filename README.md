# AuralScript

**AuralScript** converts audio files into structured text schemas that can be read and analyzed by a Large Language Model (LLM) like Claude.

The core idea: AI music generators like Suno AI produce outputs that often drift from the intended style prompt. AuralScript closes the feedback loop — extract acoustic features from the output, feed them to an LLM, and get data-driven suggestions on why the output drifted and what prompt adjustments to make.

---

## What it does

AuralScript extracts and serializes the following from any MP3 or WAV file:

| Section | What it captures |
|---|---|
| META | BPM, key, tempo stability, beat strength |
| ENERGY | RMS, dynamic range, loudness level |
| BASS | Sub vs mid-bass balance, guitar vs synth bass character |
| ONSET DENSITY | Arrangement busyness, hits per second |
| SPECTRUM | Brightness, spectral centroid, bandwidth |
| DISTORTION | High-frequency noise floor distortion estimate |
| STRUCTURE | Quiet/loud alternating pattern, verse-chorus detection |
| HARMONIC | Chord complexity, tonal stability, change rate |
| REVERB / ROOM | Decay time estimate, electronic vs organic instrument hint |
| STEREO WIDTH | Mono vs wide mix (M/S ratio) |
| TEXTURE | Harmonic/percussive balance, roughness, MFCCs |
| VOCAL | Vocal presence and character in mix |
| TRANSIENTS | Attack sharpness (Peak/Mean ratio) |
| WAVEFORM | ASCII amplitude plot of the full song |
| TIMELINE | 16-segment breakdown: energy, key, vocal, pitch confidence |
| PANNS TAGS | Neural audio tagging — genre, instrument, vocal style (optional) |

---

## Example output

```
=================================================================
AURALSCRIPT v2.1
Source:    worldgonemad.mp3
Mode:      Raw Suno Output
=================================================================

── PANNS AUDIO TAGS ──────────────────────────────────────────
  Music-relevant tags:
    0.867 █████████████████    Music
    0.115 ██                   Heavy metal
    0.090 █                    Punk rock
    0.052 █                    Angry music
  All top tags:
    0.867  Music
    0.221  Speech
    0.151  Yell
    0.090  Punk rock
    0.082  Whoop
```

---

## Installation

### Step 1 — Install Python 3.12

Download from https://www.python.org/downloads/

**Critical:** On the first installer screen, check **"Add Python to PATH"**

Verify:
```
python --version
```

### Step 2 — Install core dependencies

```
pip install librosa
pip install numpy
pip install scipy
pip install soundfile
```

### Step 3 — Install PANNs (optional but recommended)

PANNs adds neural audio tagging — genre, instrument, and vocal style classification using a model trained on 527 AudioSet sound categories.

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install panns-inference
```

Then manually download two files into `C:\Users\<yourname>\panns_data\`:

**File 1 — Class labels CSV:**
```
http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv
```
Save as: `class_labels_indices.csv`

**File 2 — Model checkpoint (~500MB):**
```
https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1
```
Save as: `Cnn14_mAP=0.431.pth`

> Note: PANNs has Windows-specific setup issues. If it doesn't work, AuralScript runs fine without it — all other sections still work.

---

## Usage

### Basic run
```
python auralscript_extract.py "your_song.mp3"
```
Output saved as: `your_song_auralscript.txt`

### Specify output path
```
python auralscript_extract.py "your_song.mp3" --out "results/song_v1.txt"
```

### DAW-processed files
If the file has been through a DAW (edited, mixed, mastered beyond the raw Suno output), add `--processed`. This adjusts thresholds for the structure detector and distortion estimator:
```
python auralscript_extract.py "your_song.mp3" --processed
```

---

## The Suno AI workflow

1. Write a style prompt in Suno AI and generate a song
2. Download the MP3
3. Run AuralScript on it
4. Fill in the SUNO TAGS section at the bottom of the output
5. Paste the full output to Claude (or another LLM)
6. Ask: *"Based on this AuralScript output and my style prompt, why did it drift and what should I change?"*
7. Iterate

### Example prompt to Claude:
```
Here is my AuralScript output and my Suno style prompt.
The song was supposed to be 90s grunge with quiet verses and loud choruses.
Based on the data, what went wrong and what tags should I adjust?

[paste AuralScript output]

Style prompt used: [paste your Suno style prompt]
```

---

## What the data tells an LLM

The key genre-separating signals discovered during development:

| Signal | Electronic | Organic/Rock |
|---|---|---|
| Decay Consistency | ~0.000 | 0.3–0.9 |
| Spectral Centroid | >4000 Hz | 2500–3500 Hz |
| PANNs tags | Electronic, Synth, EDM | Guitar, Rock, Grunge |
| Tempo Stability | ±0.0 BPM | ±0.5–2.0 BPM |
| Distortion HF/Mid | -6 to -8 dB | -10 to -14 dB |

---

## Limitations

- **ZCR roughness** is unreliable for distinguishing distorted guitar from clean synth on compressed audio
- **Pitch confidence (PConf)** is low across all segments for dense/compressed mixes — this is expected, not a bug
- **Dynamic range** can read as 0 or absurdly high if the file has near-silence at start/end — v2.1 excludes bottom 5% RMS frames to mitigate this
- **PANNs** does not support Windows natively — requires manual file download workaround
- BPM sometimes halves or doubles — if the reading seems wrong, mentally multiply or divide by 2

---

## Background

AuralScript was developed in a single session as an experiment in whether acoustic feature extraction could close the feedback loop on AI-generated music. The concept — extract features from AI music output, serialize to LLM-readable text, feed back to the AI ecosystem that generated it — appears to be novel. No prior art found for this specific closed-loop application.

---

## License

MIT License — use freely, attribution appreciated.

---

## Contributing

Issues and PRs welcome. Key areas for improvement:
- Windows-native neural audio tagging alternative to PANNs
- Similarity scorer / song library database mode
- Better distortion detection on brick-wall limited audio
- Vocal delivery classification (mumbled vs shouted vs sung)
