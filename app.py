import os, uuid, subprocess, threading, json
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB

ALLOWED = {"wav", "mp3", "ogg", "webm", "m4a", "aac"}
TMP_DIR = "temp_audio"
os.makedirs(TMP_DIR, exist_ok=True)

ESCALATION_PHRASES = [
    "cancel my account", "speak to a manager", "speak to your manager",
    "talk to a manager", "lawyer", "attorney", "sue you", "supervisor",
]


# ---------------------------------------------------------------- Speech

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


# -------------------------------------------------------------- Language

def get_language_client():
    return TextAnalyticsClient(
        endpoint=os.environ["AZURE_LANGUAGE_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_LANGUAGE_KEY"]),
    )


def decide_escalation(overall, sentences, raw_text):
    reasons = []
    if overall == "negative":
        reasons.append("overall_negative")
    if sum(1 for s in sentences if s["negative"] >= 0.80) >= 2:
        reasons.append("multiple_negative_sentences")
    lowered = raw_text.lower()
    if any(p in lowered for p in ESCALATION_PHRASES):
        reasons.append("escalation_phrase")
    return {"flagged": bool(reasons), "reasons": reasons}


def analyze_text(text):
    client = get_language_client()

    # 1. PII first: the raw text goes to exactly one Language call
    pii = client.recognize_pii_entities([text], language="en")[0]
    if pii.is_error:
        raise RuntimeError(pii.error.message)
    redacted = pii.redacted_text
    pii_items = [{"category": e.category, "confidence": round(e.confidence_score, 2)}
                 for e in pii.entities]  # entity text is not kept

    # 2. Everything else runs on the redacted text
    sent = client.analyze_sentiment([redacted], language="en")[0]
    kp = client.extract_key_phrases([redacted], language="en")[0]
    ner = client.recognize_entities([redacted], language="en")[0]
    for r in (sent, kp, ner):
        if r.is_error:
            raise RuntimeError(r.error.message)

    sentences = [{
        "text": s.text,
        "label": s.sentiment,
        "negative": round(s.confidence_scores.negative, 2),
    } for s in sent.sentences]

    return {
        "redacted_text": redacted,
        "key_phrases": list(kp.key_phrases),
        "entities": [{"text": e.text, "category": e.category} for e in ner.entities],
        "sentiment": {
            "label": sent.sentiment,
            "scores": {
                "positive": round(sent.confidence_scores.positive, 2),
                "neutral": round(sent.confidence_scores.neutral, 2),
                "negative": round(sent.confidence_scores.negative, 2),
            },
        },
        "sentences": sentences,
        "pii": {
            "redacted_count": len(pii_items),
            "categories": sorted({i["category"] for i in pii_items}),
            "items": pii_items,
        },
        "escalation": decide_escalation(sent.sentiment, sentences, text),
    }


@app.route("/analyze", methods=["POST"])
def analyze():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "JSON body must include non-empty 'text'"}), 400
    try:
        return jsonify(analyze_text(text))
    except Exception as e:
        return jsonify({"error": "Analysis failed", "detail": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)