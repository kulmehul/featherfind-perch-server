"""
FeatherFind Perch Server (PRIMARY sound-ID engine)
---------------------------------------------------
Wraps Google's Perch model (via the bioacoustics-model-zoo package)
to identify bird species from audio recordings.

WHY THIS FILE IS STRUCTURED THE WAY IT IS:
This server is deliberately split into two layers:

  1. MODEL_LOADER / MODEL  -- the ONLY place that knows which specific
     bioacoustics model is in use. Everything below this point talks
     to a generic `model.predict(files)` interface.

  2. Flask routes -- talk only to the abstraction above, never
     directly to bmz.Perch() or any model-specific detail.

This means swapping to a different model later (BirdNET, HawkEars,
a future Perch version, or something not yet released) should only
ever require changing the MODEL_LOADER section below -- nothing in
the Flask routes, response formatting, or rate-limiting logic should
need to change. See backend-birdnet/server.py for the secondary
engine, which intentionally mirrors this same structure.

MODEL CHOICE NOTE (read before changing):
We use bmz.Perch() -- the ORIGINAL Perch model -- not bmz.Perch2().
Perch2 requires TensorFlow >=2.20.0 and, per Google's own model card,
currently requires a GPU. Render's free tier has no GPU. If a CPU-
compatible Perch2 build becomes officially available and well-tested
in the future, swapping to it should be possible by changing only the
MODEL_LOADER section below.

LICENSE NOTE: Perch (both versions) is Apache 2.0 -- fully permissive,
safe for commercial/app-store distribution, no restrictions. This is
the main reason Perch is the PRIMARY engine, with BirdNET (CC BY-NC-SA,
non-commercial) kept as a secondary/comparison engine instead.

SETUP (for a parent/guardian):
  1. pip install -r requirements.txt
  2. python server.py
  3. The first run will auto-download the Perch model files --
     this only happens once, and may take a few minutes.
  4. Deploy this to a free Python host (Render.com free tier works;
     same approach as backend-birdnet/server.py).
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os
from datetime import date

app = Flask(__name__)
CORS(app)

# =================================================================
# MODEL_LOADER -- the ONLY section that should change if/when we
# swap to a different bioacoustics model. Everything below this
# block in the rest of the file is model-agnostic.
# =================================================================

print("Loading Perch model, this may take a moment on first run...")
import bioacoustics_model_zoo as bmz
MODEL = bmz.Perch()
MODEL_NAME = "Perch (Google, v1)"
print(f"{MODEL_NAME} loaded successfully.")


def run_model_prediction(audio_file_path):
    """
    Model-agnostic prediction wrapper.

    Takes a path to an audio file, returns a list of
    {"commonName": str, "scientificName": str, "confidence": float 0-100}
    dicts, sorted by confidence descending, top 3 only.

    If swapping models: as long as the new model also exposes a
    `.predict([file_path])` method returning a per-class-score
    dataframe (as every model in bioacoustics-model-zoo does), this
    function should not need to change at all -- only MODEL above.
    """
    scores_df = MODEL.predict([audio_file_path])

    # scores_df has one row per (file, start_time, end_time) window,
    # one column per species. Take the max score per species across
    # all time windows in the recording.
    max_scores = scores_df.max(axis=0)
    top = max_scores.sort_values(ascending=False).head(3)

    results = []
    for class_label, score in top.items():
        # bioacoustics-model-zoo class labels are typically formatted
        # as "Scientific name_Common Name" (same convention as BirdNET)
        if isinstance(class_label, str) and "_" in class_label:
            scientific, common = class_label.split("_", 1)
        else:
            scientific, common = "", str(class_label)

        import math
        probability = 1.0 / (1.0 + math.exp(-float(score)))
        results.append({
            "commonName": common,
            "scientificName": scientific,
            "confidence": round(probability * 100, 1),
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
