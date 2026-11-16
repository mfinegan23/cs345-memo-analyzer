import os, uuid, subprocess, threading, tempfile
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
import azure.cognitiveservices.speech as speechsdk

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB

ALLOWED = {"wav", "mp3", "ogg", "webm", "m4a", "aac"}
TMP_DIR = "temp_audio"
os.makedirs(TMP_DIR, exist_ok=True)


def to_wav(src, dst):
    """Convert any input to 16 kHz mono 16-bit WAV with ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1",
         "-sample_fmt", "s16", dst],
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError("Could not decode audio file")


def transcribe_wav(wav_path):
    speech_config = speechsdk.SpeechConfig(
        subscription=os.environ["AZURE_SPEECH_KEY"],
        region=os.environ["AZURE_SPEECH_REGION"],
    )
    speech_config.request_word_level_timestamps()
    speech_config.output_format = speechsdk.OutputFormat.Detailed
    auto_lang = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=["en-US"]
    )
    audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
        auto_detect_source_language_config=auto_lang,
    )

    import json
    done = threading.Event()
    texts, words, confs = [], [], []
    state = {"error": None}

    def on_recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            detail = json.loads(evt.result.json)
            best = detail["NBest"][0]
            texts.append(best["Display"])
            confs.append(best["Confidence"])
            for w in best.get("Words", []):
                words.append({
                    "word": w["Word"],
                    "offset": w["Offset"] / 1e7,
                    "duration": w["Duration"] / 1e7,
                    "confidence": w.get("Confidence"),
                })

    def on_canceled(evt):
        if evt.cancellation_details.reason == speechsdk.CancellationReason.Error:
            state["error"] = evt.cancellation_details.error_details
        done.set()

    recognizer.recognized.connect(on_recognized)
    recognizer.session_stopped.connect(lambda evt: done.set())
    recognizer.canceled.connect(on_canceled)

    recognizer.start_continuous_recognition_async().get()
    done.wait(timeout=120)
    recognizer.stop_continuous_recognition_async().get()

    if state["error"]:
        raise RuntimeError(state["error"])

    duration = (words[-1]["offset"] + words[-1]["duration"]) if words else 0.0
    return {
        "transcript": " ".join(texts),
        "language": "en-US",
        "duration_seconds": round(duration, 2),
        "confidence": round(sum(confs) / len(confs), 3) if confs else 0.0,
        "words": words,
    }


@app.route("/transcribe", methods=["POST"])
def transcribe():
    f = request.files.get("audio")
    if not f or not f.filename:
        return jsonify({"error": "No audio file provided (field name: audio)"}), 400
    ext = f.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported audio format: .{ext}"}), 415

    uid = uuid.uuid4().hex
    src = os.path.join(TMP_DIR, f"{uid}_in.{ext}")
    wav = os.path.join(TMP_DIR, f"{uid}_16k.wav")
    try:
        f.save(src)
        try:
            to_wav(src, wav)
        except ValueError as e:
            return jsonify({"error": str(e)}), 415
        return jsonify(transcribe_wav(wav))
    except Exception as e:
        return jsonify({"error": "Transcription failed", "detail": str(e)}), 500
    finally:
        for p in (src, wav):
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)