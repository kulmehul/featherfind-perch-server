"""
FeatherFind Perch Server (PRIMARY sound-ID engine)
---------------------------------------------------
Wraps Google's Perch model, loaded DIRECTLY from TensorFlow Hub,
to identify bird species from audio recordings.

WHY THIS FILE IS STRUCTURED THE WAY IT IS:
This server is deliberately split into two layers:

  1. MODEL_LOADER / MODEL  -- the ONLY place that knows which specific
     bioacoustics model is in use and how it's loaded. Everything
     below this point talks to a generic prediction interface.

  2. Flask routes -- talk only to the abstraction above, never
     directly to the TF-Hub model object.

This means swapping to a different model later should only ever
require changing the MODEL_LOADER section below -- nothing in the
Flask routes, response formatting, or rate-limiting logic should
need to change. See backend-birdnet/server.py for the secondary
engine, which intentionally mirrors this same structure.

WHY DIRECT TF-HUB LOADING (not the bioacoustics-model-zoo package):
We originally used the `bioacoustics-model-zoo` package's `bmz.Perch()`
wrapper. It works, but it has `opensoundscape` as a hard (non-optional)
dependency, which in turn pulls in a large, unrelated set of packages
(Jupyter notebook server components, etc.) that we never use. On
Render's free tier, this made installs slow and pushed the deploy past
Render's port-detection timeout -- a real problem hit and diagnosed
during this project's deployment (see ARCHITECTURE_HANDOFF.md section
3.2 for the full story). Loading directly from TensorFlow Hub avoids
this entirely: only `tensorflow`, `tensorflow_hub`, and `librosa` (for
audio loading/resampling) are needed. This is the same underlying
Perch model, same license, same accuracy -- just a lighter path to it.

MODEL CHOICE NOTE (read before changing):
This is Perch v1 (`bird-vocalization-classifier/1` on TF-Hub) -- the
ORIGINAL Perch model, not Perch2. Perch2 currently requires a GPU per
Google's own model card. Render's free tier has no GPU. Do not swap to
Perch2 here without first re-confirming a CPU-compatible build is
officially available and well-tested.

LICENSE NOTE: Perch is Apache 2.0 -- fully permissive, safe for
commercial/app-store distribution, no restrictions. This is the main
reason Perch is the PRIMARY engine, with BirdNET (CC BY-NC-SA,
non-commercial) kept as a secondary/comparison engine instead.

SETUP (for a parent/guardian):
  1. pip install -r requirements.txt
  2. python server.py
  3. The first run downloads the Perch model (~80MB) from TF-Hub; the
     associated species labels list is resolved locally from the cached
     assets without any additional Kaggle download requests.
  4. Deploy this to a free Python host (Render.com free tier works).
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os
import math
from datetime import date

app = Flask(__name__)
CORS(app)

# =================================================================
# MODEL_LOADER -- the ONLY section that should change if/when we
# swap to a different bioacoustics model. Everything below this
# block in the rest of the file is model-agnostic.
# =================================================================

print("Loading Perch model from TensorFlow Hub, this may take a moment on first run...")
import numpy as np
import tensorflow_hub as hub
import csv

MODEL = hub.load("https://tfhub.dev/google/bird-vocalization-classifier/1")
MODEL_NAME = "Perch (Google, v1, direct TF-Hub)"

# The model's label list (eBird species codes, one per output column)
# is cached locally by TensorFlow Hub when the model is loaded.
# We resolve its path and read it directly from the local assets folder.
LABELS = []
try:
    model_path = hub.resolve("https://tfhub.dev/google/bird-vocalization-classifier/1")
    labels_path = os.path.join(model_path, "assets", "label.csv")
    with open(labels_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        LABELS = [row["ebird2021"] for row in reader if row.get("ebird2021")]
    print(f"Loaded {len(LABELS)} species labels.")
except Exception as e:
    print(f"WARNING: could not load labels file ({e}). Falling back to raw index numbers as labels.")
    LABELS = []

print(f"{MODEL_NAME} loaded successfully.")


def run_model_prediction(audio_file_path):
    """
    Model-agnostic prediction wrapper.

    Takes a path to an audio file, returns a list of
    {"commonName": str, "scientificName": str, "confidence": float 0-100}
    dicts, sorted by confidence descending, top 3 only.
    """
    import librosa

    # Load audio, resampled to 32kHz mono, as the model requires.
    waveform, _ = librosa.load(audio_file_path, sr=32000, mono=True)

    # The model expects 5-second (160,000 sample) windows. Pad short
    # clips with silence; for longer clips, just use the first 5
    # seconds (good enough for a hobby app; not doing multi-window
    # analysis here, unlike the bioacoustics-model-zoo wrapper did).
    target_len = 5 * 32000
    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        waveform = waveform[:target_len]

    waveform = waveform.astype(np.float32)[np.newaxis, :]

    logits, _embeddings = MODEL.infer_tf(waveform)
    logits = logits.numpy()[0]  # shape: (num_classes,)

    # Apply sigmoid: Perch's raw outputs are uncalibrated logits, not
    # probabilities. Without this, confidence values can be far
    # outside a sane 0-100% range (this was caught and fixed during
    # testing -- see ARCHITECTURE_HANDOFF.md section 3.2).
    probabilities = 1.0 / (1.0 + np.exp(-logits))

    top_indices = np.argsort(probabilities)[::-1][:3]

    results = []
    for idx in top_indices:
        label = LABELS[idx] if idx < len(LABELS) else f"species_{idx}"
        if "_" in label:
            scientific, common = label.split("_", 1)
        else:
            scientific, common = "", label

        results.append({
            "commonName": common,
            "scientificName": scientific,
            "confidence": round(float(probabilities[idx]) * 100, 1),
        })

    return results


# =================================================================
# Everything below this line is model-agnostic and intentionally
# mirrors backend-birdnet/server.py's structure.
# =================================================================

MAX_REQUESTS_PER_DAY = 100
request_count = {"date": None, "count": 0}


def check_and_increment_rate_limit():
    today = str(date.today())
    if request_count["date"] != today:
        request_count["date"] = today
        request_count["count"] = 0
    if request_count["count"] >= MAX_REQUESTS_PER_DAY:
        return False
    request_count["count"] += 1
    return True


@app.route("/identify-sound", methods=["POST"])
def identify_sound():
    if not check_and_increment_rate_limit():
        return jsonify({
            "error": "Daily identification limit reached. Please try again tomorrow.",
            "matches": []
        }), 429

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided.", "matches": []}), 400

    audio_file = request.files["audio"]

    suffix = os.path.splitext(audio_file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        matches = run_model_prediction(tmp_path)

        if not matches:
            return jsonify({"matches": [], "error": "No bird sound clearly detected in this recording."})

        return jsonify({"matches": matches})

    except Exception as e:
        return jsonify({"error": f"Could not analyze audio: {str(e)}", "matches": []}), 500

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
