import os, uuid, subprocess, threading, json, base64, wave
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient, PiiEntityCategory
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
PII_CATEGORIES = [
    PiiEntityCategory.PERSON, PiiEntityCategory.PHONE_NUMBER,
    PiiEntityCategory.EMAIL, PiiEntityCategory.ADDRESS,
    PiiEntityCategory.CREDIT_CARD_NUMBER,
]
VOICES = {"jenny": "en-US-JennyNeural", "guy": "en-US-GuyNeural",
          "aria": "en-US-AriaNeural"}


# ---------------------------------------------------------------- Speech

def cleanup(*paths):
    for p in paths:
        if p and os.path.exists(p):
            os.remove(p)


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

    # Billed audio length: measure the WAV, not the last word's end
    with wave.open(wav_path, "rb") as w:
        duration = w.getnframes() / w.getframerate()

    return {
        "transcript": " ".join(texts),
        "language": "en-US",
        "duration_seconds": round(duration, 2),
        "confidence": round(sum(confs) / len(confs), 3) if confs else 0.0,
        "words": words,
    }


def save_and_convert(f):
    """Save upload, convert to WAV. Returns (src, wav); caller must clean up."""
    ext = f.filename.rsplit(".", 1)[-1].lower()
    uid = uuid.uuid4().hex
    src = os.path.join(TMP_DIR, f"{uid}_in.{ext}")
    wav = os.path.join(TMP_DIR, f"{uid}_16k.wav")
    try:
        f.save(src)
        to_wav(src, wav)
    except Exception:
        cleanup(src, wav)  # don't leak temp files if conversion fails
        raise
    return src, wav


def check_upload():
    """Returns (file, error_response)."""
    f = request.files.get("audio")
    if not f or not f.filename:
        return None, (jsonify({"error": "No audio file provided (field name: audio)"}), 400)
    ext = f.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED:
        return None, (jsonify({"error": f"Unsupported audio format: .{ext}"}), 415)
    return f, None


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Internal/testing endpoint. Returns the raw transcript."""
    f, err = check_upload()
    if err:
        return err
    src = wav = None
    try:
        try:
            src, wav = save_and_convert(f)
        except ValueError as e:
            return jsonify({"error": str(e)}), 415
        return jsonify(transcribe_wav(wav))
    except Exception as e:
        return jsonify({"error": "Transcription failed", "detail": type(e).__name__}), 500
    finally:
        cleanup(src, wav)


# -------------------------------------------------------------- Language

def get_language_client():
    return TextAnalyticsClient(
        endpoint=os.environ["AZURE_LANGUAGE_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_LANGUAGE_KEY"]),
    )


def decide_escalation(overall, sentences, text):
    reasons = []
    if overall == "negative":
        reasons.append("overall_negative")
    if sum(1 for s in sentences if s["negative"] >= 0.80) >= 2:
        reasons.append("multiple_negative_sentences")
    lowered = text.lower()
    if any(p in lowered for p in ESCALATION_PHRASES):
        reasons.append("escalation_phrase")
    return {"flagged": bool(reasons), "reasons": reasons}


def analyze_text(text):
    client = get_language_client()

    # 1. PII first: the raw text goes to exactly one Language call
    pii = client.recognize_pii_entities(
        [text], language="en", categories_filter=PII_CATEGORIES)[0]
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
        "escalation": decide_escalation(sent.sentiment, sentences, redacted),
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
        return jsonify({"error": "Analysis failed", "detail": type(e).__name__}), 500


# ------------------------------------------------------------------- TTS

def build_summary(analysis):
    """Spoken summary built only from the redacted analysis."""
    esc = analysis["escalation"]
    parts = []
    if esc["flagged"]:
        parts.append("This call was flagged for escalation.")
    else:
        parts.append("This call was not flagged for escalation.")
    parts.append(f"The overall tone was {analysis['sentiment']['label']}.")
    n = analysis["pii"]["redacted_count"]
    if n:
        parts.append(f"{n} piece{'s' if n != 1 else ''} of personal information "
                     f"{'were' if n != 1 else 'was'} redacted.")
    # skip phrases containing redaction masks so TTS doesn't read "star star"
    phrases = [p for p in analysis["key_phrases"] if "*" not in p][:3]
    if phrases:
        parts.append("Key topics were " + ", ".join(phrases) + ".")
    return " ".join(parts)


def synthesize(text, voice="en-US-JennyNeural"):
    cfg = speechsdk.SpeechConfig(
        subscription=os.environ["AZURE_SPEECH_KEY"],
        region=os.environ["AZURE_SPEECH_REGION"],
    )
    cfg.speech_synthesis_voice_name = voice
    cfg.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
    synth = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=None)
    result = synth.speak_text_async(text).get()
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        details = speechsdk.SpeechSynthesisCancellationDetails.from_result(result)
        raise RuntimeError(f"TTS failed: {details.error_details}")
    return {
        "audio_base64": base64.b64encode(result.audio_data).decode("ascii"),
        "char_count": len(text),
    }


# --------------------------------------------------------------- Process

@app.route("/process", methods=["POST"])
def process():
    f, err = check_upload()
    if err:
        return err
    voice = VOICES.get(request.form.get("voice", "jenny"), VOICES["jenny"])
    src = wav = None
    try:
        try:
            src, wav = save_and_convert(f)
        except ValueError as e:
            return jsonify({"error": str(e)}), 415

        stt = transcribe_wav(wav)                 # raw transcript stays in memory
        if not stt["transcript"].strip():
            return jsonify({"error": "No speech detected in audio"}), 422

        analysis = analyze_text(stt["transcript"])
        summary = build_summary(analysis)
        tts = synthesize(summary, voice)

        return jsonify({
            "transcript": analysis["redacted_text"],   # redacted only
            "language": stt["language"],
            "duration_seconds": stt["duration_seconds"],
            "confidence": stt["confidence"],
            "sentiment": analysis["sentiment"],
            "sentences": analysis["sentences"],
            "key_phrases": analysis["key_phrases"],
            "entities": analysis["entities"],
            "pii": {"redacted_count": analysis["pii"]["redacted_count"],
                    "categories": analysis["pii"]["categories"]},
            "escalation": analysis["escalation"],
            "summary_text": summary,
            "audio_base64": tts["audio_base64"],
        })
    except Exception as e:
        return jsonify({"error": "Processing failed", "detail": type(e).__name__}), 500
    finally:
        cleanup(src, wav)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="127.0.0.1", port=port)