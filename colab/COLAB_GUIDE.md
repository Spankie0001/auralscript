# How to Analyze a Song's Structure Using AuralScript-Lite in Google Colab

**No installs. No Python knowledge. Free. Works on Mac, PC, Chromebook, phone, tablet — anywhere you have a browser.**

This guide is for people who want to see the spectral structure of any song (BPM, energy curve, brightness over time, bass-vs-mid balance, where the climax actually is) but don't want to install anything on their computer.

---

## What you'll get out of this

For any mp3 you upload, you'll see:

- Global BPM
- An 8-section spectral table showing how the song's energy and brightness change over time
- An energy curve (text bar chart) showing exactly where the peak is
- The option to define your own custom sections matching the song's real structure

This is useful for:

- **Suno producers** wanting to verify their AI-generated cover actually matches the source's energy arc
- **Cover artists** wanting to know what makes a reference track work before trying to recreate it
- **Anyone curious** about why a song feels the way it does — the data shows you the producer's structural moves

---

## Setup, one time only — about 90 seconds

### Step 1: get a free Google account if you don't have one

If you already use Gmail, Google Drive, or YouTube — you're done. Skip ahead.

If not: go to https://accounts.google.com/signup and make one. Takes 60 seconds.

### Step 2: open Google Colab

Go to https://colab.research.google.com in your browser. Sign in with your Google account if prompted.

You'll see a welcome screen with a popup that says "Open notebook." Close it for now (X in the corner).

### Step 3: open the AuralScript-Lite notebook

Two ways to do this, pick whichever's easier for you:

**Option A — Upload the file:**

1. Download `AuralScript_Lite.ipynb` from wherever it's hosted (the link the person who shared this with you provided).
2. In Colab, click **File → Upload notebook**.
3. Pick the `.ipynb` file you just downloaded.

**Option B — Open from GitHub (if it's hosted on GitHub):**

1. In Colab, click **File → Open notebook**.
2. Click the **GitHub** tab.
3. Paste the GitHub URL of the notebook.
4. Hit search, then click the result.

Either way, the notebook will open and you'll see a bunch of cells with text and code.

---

## Running it — about 2 minutes per song

### Step 4: install ffmpeg (cell 1)

Find the first cell that has code in it (it should say something like `!apt-get install -y ffmpeg`).

Click anywhere inside it. You'll see a play button ▶ on the left side of the cell.

Click the play button (or press **Shift+Enter** on your keyboard).

The first time you do this in a session, Colab will pop up a warning saying "this notebook was not authored by Google." Click **Run anyway**. (This warning shows up for every notebook that isn't from Google itself. It's normal.)

Wait about 5-10 seconds. You should see:

```
✓ ffmpeg ready
✓ numpy 1.x.x
✓ scipy 1.x.x
```

That's the only "install" step. Everything else is just running cells.

### Step 5: upload your song (cell 2)

Click the next code cell (the one that says `from google.colab import files`).

Hit the play button (or Shift+Enter).

A file picker will appear. **Choose Files** → pick your mp3.

Wait for the upload (a 5 MB mp3 takes about 5 seconds).

You should see:
```
✓ Uploaded: yoursong.mp3 (4.27 MB)
```

**Note:** Keep files under 25 MB. Standard mp3s (128-320 kbps) are well under this for songs up to about 25 minutes. If you have a wav or flac, convert it to mp3 first or it might be too big.

### Step 6: run the rest of the cells in order

Either:

- Click each remaining code cell and hit the play button, **or**
- Click **Runtime → Run all** in the top menu to run everything at once.

The cells will run in order. Each takes a few seconds. Total time for a 4-minute song: about 30 seconds for all the analysis to complete.

You'll see output appear under each cell:

- **Step 3** shows the file converted to wav
- **Step 4** confirms the analysis tool is loaded
- **Step 5** prints the BPM and the 8-section default analysis table
- **Step 6** prints the energy curve with bar chart and tells you where the peak is
- **Step 7** prints custom-section analysis (you'll edit this — see next step)

### Step 7: define your own sections (the useful part)

This is where the analysis goes from "interesting" to "actually useful."

Scroll to the cell labeled **Step 7 — Define your own sections**. You'll see a Python list that looks like this:

```python
MY_SECTIONS = [
    {'name': 'INTRO',     'start': 0.0,   'end': 25.0,  'note': 'spoken word + pad'},
    {'name': 'ARP_IN',    'start': 25.0,  'end': 50.0,  'note': 'arpeggio enters'},
    {'name': 'DRUMS_IN',  'start': 50.0,  'end': 100.0, 'note': 'drums establish'},
    ...
]
```

Edit the numbers and labels to match your actual song. Look at the energy curve from Step 6 to find the structural transitions, then plug those timestamps in here.

For example, if you're analyzing your own Suno output and you wrote it with these sections:

- 0:00–0:15 intro
- 0:15–0:30 arp enters
- 0:30–1:00 drums come in
- 1:00–2:00 build
- 2:00–3:00 climax
- 3:00–3:45 outro

Your `MY_SECTIONS` should look like:

```python
MY_SECTIONS = [
    {'name': 'INTRO',    'start': 0.0,   'end': 15.0,  'note': 'pad only'},
    {'name': 'ARP_IN',   'start': 15.0,  'end': 30.0,  'note': 'arp enters'},
    {'name': 'DRUMS',    'start': 30.0,  'end': 60.0,  'note': 'drums in'},
    {'name': 'BUILD',    'start': 60.0,  'end': 120.0, 'note': 'filter opens'},
    {'name': 'CLIMAX',   'start': 120.0, 'end': 180.0, 'note': 'peak'},
    {'name': 'OUTRO',    'start': 180.0, 'end': 225.0, 'note': 'collapse'},
]
```

Save and run the cell again (Shift+Enter or play button). The table now shows analysis for your real song structure.

### Step 8: read the numbers

The header in Step 8 of the notebook explains what each column means. Quick reference:

- **RMS (dB)** — loudness. -25 is loud, -40 is quiet
- **CENT (Hz)** — spectral brightness. Below 1500 = dark, above 2500 = bright
- **BASS-MID** — positive = bassy mix, negative = mid/treble forward
- **ONS/s** — onsets per second, how busy the section is

---

## Common questions

**"It says the runtime disconnected."**

Colab kills sessions that sit idle. Just hit play on the install cell again — it re-runs in 5 seconds. Your uploaded mp3 stays in the session.

**"My file is too big."**

Compress to mp3 first. There are free converters online (CloudConvert, etc.). 128 kbps mp3 is plenty for analysis.

**"The BPM is wrong."**

The simple autocorrelation BPM detector is fragile. It works well on steady techno/rock/pop with prominent drums. It struggles on sparse arrangements, complex meters, or rubato music. For those, use the main AuralScript with librosa's `beat_track()`.

**"Can I save the results?"**

Yes — copy-paste the output into a note app, or take screenshots. You can also right-click the cell output and **Save image as** if you want to save the bar chart visually. For a more permanent record, paste the table into a markdown file alongside your session doc.

**"Do I need to re-upload every time?"**

If you keep the same Colab session open, no. If you close and come back, yes — Colab is ephemeral and clears uploads when the session ends.

**"Is my song uploaded somewhere permanent?"**

No. Colab sessions are sandboxed and wiped when the session ends. Your mp3 isn't stored anywhere persistent and isn't shared with anyone. Google's standard privacy terms apply — they're not training models on your Colab uploads as far as anyone has documented, but if you're paranoid about it, use songs you don't mind being seen.

---

## What this won't do

- Neural genre tagging (the full AuralScript with PANNs does that)
- Lyrics transcription (use Whisper directly for that)
- Stereo width analysis (mono only here)
- Reliable BPM on sparse-drum / non-4/4 material

For all of those, the canonical full AuralScript at https://github.com/Spankie0001/auralscript is the right tool. It needs Python and librosa installed locally, but it does the heavy stuff.

This Colab is the lightweight option — for people who just want to see the structure of a song quickly, in a browser, with no setup.

---

## How to share this with someone else

If you want to share this with another producer:

1. Upload the `AuralScript_Lite.ipynb` file to your Google Drive, or host it on GitHub.
2. Send them this guide along with a link to the notebook.
3. They follow Steps 1-8 above.

You can also create a permanent shareable Colab link:

1. Open the notebook in Colab.
2. Click **Share** in the top right.
3. Change access to "Anyone with the link → Viewer."
4. Copy the link and send it.

When they open it, they'll get a copy they can edit and run themselves. Their copy doesn't change yours.

---

Built collaboratively in a Claude conversation with Jason Alvey (Spankie0001) for the Suno community. CC0 / public domain. Modify and redistribute however you want.
