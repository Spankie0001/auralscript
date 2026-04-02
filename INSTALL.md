# AuralScript — Detailed Installation Guide
## Windows 10/11, Python 3.12

---

## Prerequisites

- Windows 10 or 11
- Internet connection
- ~1GB free disk space for core install
- ~1GB additional for Whisper language detection (optional but recommended)
- ~2.5GB additional for PANNs neural tagging (optional)

---

## Full dependency list

| Library | Required | Purpose |
|---|---|---|
| librosa | ✅ Core | Acoustic feature extraction |
| numpy | ✅ Core | Numeric processing |
| scipy | ✅ Core | Bandpass filtering |
| soundfile | ✅ Core | MP3/WAV loading |
| torch (CPU) | Optional | Required for PANNs and Whisper |
| panns-inference | Optional | Neural genre/instrument tagging |
| faster-whisper | Optional | Language detection + lyrics |

---

## Step 1 — Install Python 3.12

1. Go to: https://www.python.org/downloads/release/python-3120/
2. Scroll down to **"Windows installer (64-bit)"** and download it
3. Run the installer
4. **CRITICAL — check this box on the first screen:**
   ```
   [✓] Add Python to PATH
   ```
   If you miss this, nothing works from the command line.
5. Click **"Install Now"**

**Verify the install worked:**

Open Command Prompt (search "cmd" in Start menu) and type:
```
python --version
```
You should see: `Python 3.12.x`

---

## Step 2 — Create your working folder

```
mkdir C:\AuralScript
```

Save `auralscript_extract.py` into this folder.

---

## Step 3 — Install core Python libraries

Open Command Prompt and run these one at a time:

```
pip install librosa
pip install numpy
pip install scipy
pip install soundfile
```

**Verify librosa installed correctly:**
```
python -c "import librosa; print('librosa OK')"
```

### Known issue: numba warnings

You may see `NumbaDeprecationWarning` — harmless, ignore it or:
```
pip install numba
```

---

## Step 4 — Test the script (core only)

Put any MP3 in your C:\AuralScript folder and run:
```
cd C:\AuralScript
python auralscript_extract.py "your_song.mp3"
```

You should see output ending with:
```
[AuralScript] Saved to: your_song_auralscript.txt
```

---

## Step 5 — Install PyTorch (required for PANNs and Whisper)

Both PANNs and Whisper require PyTorch. Install the CPU version:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

This is a large download (~200MB). Let it finish.

---

## Step 6 — Install Whisper (recommended)

Whisper detects the language of the song and transcribes lyrics up to 4000 characters.
First run downloads the model (~465MB) automatically.

```
pip install faster-whisper
```

**Note:** On first run you may see:
```
Warning: You are sending unauthenticated requests to the HF Hub...
```
This is harmless — ignore it. The model downloads fine without an account.

---

## Step 7 — Install PANNs (optional)

PANNs adds neural audio tagging — identifies genres, instruments, and vocal styles
from 527 AudioSet categories. Requires manual file downloads.

**7a — Install PANNs**

```
pip install panns-inference
```

**7b — Create the PANNs data folder**

```
mkdir C:\Users\<yourname>\panns_data
```

Replace `<yourname>` with your actual Windows username.

**7c — Download the class labels file**

Open your browser and go to:
```
http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv
```

Save the file to: `C:\Users\<yourname>\panns_data\class_labels_indices.csv`

**7d — Download the model checkpoint (~500MB)**

```
https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1
```

Save to: `C:\Users\<yourname>\panns_data\`
File must be named exactly: `Cnn14_mAP=0.431.pth`

**7e — Verify PANNs works**

```
python -c "import panns_inference; print('PANNs OK')"
```

---

## Running the script

### Standard run
```
python auralscript_extract.py "your_song.mp3"
```

### Specify output file
```
python auralscript_extract.py "song.mp3" --out "results\song_v1.txt"
```

### DAW-processed files
Use this flag if the file has been edited in a DAW:
```
python auralscript_extract.py "song.mp3" --processed
```

### Expected processing time (3-minute song, CPU only)
- Core analysis: ~30 seconds
- + PANNs: ~60 seconds
- + Whisper: ~60-90 seconds
- Total with all features: ~3-4 minutes

---

## Troubleshooting

**"python is not recognized"**
Python isn't in PATH. Re-run the installer and check the PATH box.

**"No module named librosa"**
Run: `pip install librosa`

**"No module named soundfile" or MP3 errors**
Run: `pip install soundfile`

**numba errors on import**
Run: `pip install numba` or ignore — script works without it.

**PANNs FileNotFoundError on class_labels_indices.csv**
Check: `C:\Users\<yourname>\panns_data\class_labels_indices.csv`

**PANNs FileNotFoundError on .pth file**
Must be exactly: `Cnn14_mAP=0.431.pth` (with equals sign, no quotes)

**Whisper HuggingFace warning**
```
Warning: You are sending unauthenticated requests to the HF Hub...
```
Harmless — ignore it. Model downloads fine without an account.

**BPM reads as half or double the real tempo**
Known librosa issue. Mentally multiply or divide by 2.

---

## File structure after full install

```
C:\AuralScript\
    auralscript_extract.py      <- the script
    your_song.mp3               <- your audio files
    your_song_auralscript.txt   <- generated outputs

C:\Users\<you>\panns_data\
    class_labels_indices.csv    <- PANNs labels (527 AudioSet categories)
    Cnn14_mAP=0.431.pth         <- PANNs model (~500MB)

C:\Users\<you>\AppData\Local\huggingface\  (auto-created)
    faster-whisper-small\       <- Whisper model (~465MB, auto-downloaded)
```

---

## What to paste to Claude

After running AuralScript, paste the full output .txt to Claude along with:

1. Your Suno style prompt
2. Your exclude tags
3. How the song actually sounded vs what you intended

Ask Claude to identify genre drift, suggest prompt adjustments, or compare
against a previous run.


---

## Prerequisites

- Windows 10 or 11
- Internet connection
- ~1GB free disk space for core install
- ~2.5GB additional for PANNs neural tagging (optional)

---

## Step 1 — Install Python 3.12

1. Go to: https://www.python.org/downloads/release/python-3120/
2. Scroll down to **"Windows installer (64-bit)"** and download it
3. Run the installer
4. **CRITICAL — check this box on the first screen:**
   ```
   [✓] Add Python to PATH
   ```
   If you miss this, nothing works from the command line.
5. Click **"Install Now"**

**Verify the install worked:**

Open Command Prompt (search "cmd" in Start menu) and type:
```
python --version
```
You should see: `Python 3.12.x`

If you see `command not found` or `not recognized`:
- Re-run the installer
- Choose "Modify"
- Check the PATH option
- Finish

---

## Step 2 — Create your working folder

```
mkdir C:\AuralScript
```

Save `auralscript_extract.py` into this folder.

---

## Step 3 — Install core Python libraries

Open Command Prompt and run these one at a time:

```
pip install librosa
pip install numpy
pip install scipy
pip install soundfile
```

Each one downloads and installs automatically. You'll see a lot of output — that's normal.

**If you get a permission error:**
```
pip install librosa --user
```

**Verify librosa installed correctly:**
```
python -c "import librosa; print('librosa OK')"
```

### Known issue: numba warnings

Librosa optionally uses numba for speed. You may see warnings like:
```
NumbaDeprecationWarning: ...
```
These are harmless. The script works fine without numba. If they bother you:
```
pip install numba
```

---

## Step 4 — Test the script (no PANNs)

Put any MP3 in your C:\AuralScript folder and run:
```
cd C:\AuralScript
python auralscript_extract.py "your_song.mp3"
```

You should see output like:
```
[AuralScript] Loading: your_song.mp3
[AuralScript] Duration: 180.0s  Sample rate: 48000Hz
[AuralScript] Saved to: your_song_auralscript.txt
```

The output file appears in the same folder as your MP3. Open it with any text editor.

---

## Step 5 — Install PANNs (optional)

PANNs adds neural audio tagging — the most powerful genre/instrument identification feature. It requires PyTorch and a ~500MB model file.

**5a — Install PyTorch (CPU version)**

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

This is a large download (~200MB). Let it finish.

**5b — Install PANNs**

```
pip install panns-inference
```

**5c — Create the PANNs data folder**

```
mkdir C:\Users\<yourname>\panns_data
```

Replace `<yourname>` with your actual Windows username (the folder name in C:\Users\)

**5d — Download the class labels file**

Open your browser and go to:
```
http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv
```

Save the file to: `C:\Users\<yourname>\panns_data\class_labels_indices.csv`

**5e — Download the model checkpoint**

Go to:
```
https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1
```

This is ~500MB. Save it to: `C:\Users\<yourname>\panns_data\`

The file must be named exactly: `Cnn14_mAP=0.431.pth`

**5f — Verify PANNs works**

```
python -c "import panns_inference; print('PANNs OK')"
```

If this errors with `FileNotFoundError`, check that both files are in the right folder with the right names.

---

## Step 6 — Run with PANNs

```
cd C:\AuralScript
python auralscript_extract.py "your_song.mp3"
```

First run will print:
```
[AuralScript] Running PANNs audio tagging...
Checkpoint path: C:\Users\<you>\panns_data\Cnn14_mAP=0.431.pth
Using CPU.
```

PANNs adds ~30-60 seconds to processing time on CPU. Subsequent runs are faster.

---

## Running the script

### Standard run
```
python auralscript_extract.py "C:\path\to\song.mp3"
```

### Specify output file
```
python auralscript_extract.py "song.mp3" --out "results\song_v1.txt"
```

### DAW-processed files
Use this flag if the file has been edited in a DAW beyond the raw Suno output:
```
python auralscript_extract.py "song.mp3" --processed
```

---

## Troubleshooting

**"python is not recognized"**
Python isn't in PATH. Re-run the installer and check the PATH box.

**"No module named librosa"**
Run: `pip install librosa`

**"No module named soundfile" or MP3 errors**
Run: `pip install soundfile`

**numba errors on import**
Run: `pip install numba` or ignore — script works without it.

**PANNs FileNotFoundError on class_labels_indices.csv**
The file is in the wrong place or named incorrectly.
Check: `C:\Users\<yourname>\panns_data\class_labels_indices.csv`

**PANNs FileNotFoundError on .pth file**
The model checkpoint is missing or misnamed.
Must be exactly: `Cnn14_mAP=0.431.pth` (with equals sign, no quotes)

**Script is slow**
PANNs on CPU takes 30-60 seconds. pyin pitch analysis on 16 segments also adds time.
A 3-minute song typically takes 2-4 minutes total to process.

**BPM reads as half or double the real tempo**
Known librosa issue. Mentally multiply or divide by 2.

**Dynamic range reads as 0.0 dB**
File has near-silence at start or end. The script excludes bottom 5% RMS frames
but very short files or files with long silence can still trigger this.

---

## File structure after install

```
C:\AuralScript\
    auralscript_extract.py      <- the script
    your_song.mp3               <- your audio files
    your_song_auralscript.txt   <- generated outputs

C:\Users\<you>\panns_data\
    class_labels_indices.csv    <- PANNs labels (527 AudioSet categories)
    Cnn14_mAP=0.431.pth         <- PANNs model (~500MB)
```

---

## What to paste to Claude

After running AuralScript, open the output .txt file and paste its full contents to Claude along with:

1. Your Suno style prompt
2. Your exclude tags
3. How the song actually sounded vs what you intended

Ask Claude to identify genre drift, suggest prompt adjustments, or compare against a previous run.
