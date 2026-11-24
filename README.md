# Customer Call Reviewer

CS 345 semester project (Wake Forest). A support manager uploads or records a customer call. The app transcribes it (Azure AI Speech), redacts personal information and analyzes tone (Azure AI Language), flags the call for escalation with reasons, and reads a short summary aloud (neural text-to-speech). Every call is traced and measured in Azure Application Insights.

Live app: `https://<APP_NAME>.azurewebsites.net` (fill in after deployment)

## Architecture

```
browser (record / upload)
   |  POST /process (audio)
   v
Flask on App Service (gunicorn)
   1. ffmpeg -> 16 kHz mono WAV
   2. stage.speech_to_text      Azure AI Speech, continuous recognition   (raw transcript stays in memory)
   3. stage.language_analysis   Azure AI Language
        stage.pii_redaction     PII detection first
        sentiment, key phrases, entities on the REDACTED text only
        escalation rules -> flagged + reasons
   4. stage.text_to_speech      Azure neural TTS (en-US-JennyNeural / Guy / Aria), MP3 base64
   5. metrics, spans, event -> Azure Application Insights (OpenTelemetry)
```

Region: everything is in `northcentralus` because the Azure for Students policy only allows canadacentral, northcentralus, swedencentral, belgiumcentral and denmarkeast (the handout uses eastus).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Single-page frontend (recorder, upload, results) |
| POST | `/process` | Main endpoint. Form field `audio` (+ optional `voice`: jenny, guy, aria). Returns the redacted transcript, sentiment per sentence, key phrases, entities, PII summary, `escalation {flagged, reasons}`, `summary_text`, `audio_base64`. 415 for unsupported or undecodable audio, 422 when no speech is found. |
| POST | `/analyze` | JSON `{text}`. Same analysis as `/process` without audio. Returns redacted text only. |
| POST | `/transcribe` | Internal and test only. Returns the RAW transcript and word timings. Deployed: needs header `X-Internal-Token`. Locally: loopback only. |
| GET | `/telemetry-summary` | Session averages, flagged share, flag reasons, p95 latency, per-call metrics (no transcript text). |

## Escalation rules

A call is flagged when any rule fires (the reasons are listed in the response):

1. `overall_negative`: overall sentiment is negative.
2. `multiple_negative_sentences`: at least `NEG_SENTENCE_COUNT` (2) sentences have a negative score of at least `NEG_SENTENCE_THRESHOLD` (0.80).
3. `escalation_phrase`: the text contains a phrase such as "cancel my account", "speak to a manager" or "lawyer".

## Data handling

Nothing is stored permanently. Uploaded audio is a temp file deleted after each request. Only the redacted transcript and metrics are kept in an in-memory session log (lost on restart). PII detection runs before every other analysis. The raw transcript never appears in responses from `/process` or `/analyze`, logs, spans, metrics or exception messages (exceptions are recorded by type name only).

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg            # macOS
cp .env.example .env           # then fill in your own values; never commit .env
python app.py                  # http://127.0.0.1:5001  (port 5000 is used by AirPlay on macOS)
```

## Provision Azure resources (Part A)

```bash
az login
az group create --name cs345-speech-rg --location northcentralus
az cognitiveservices account create --name cs345-speech --resource-group cs345-speech-rg \
  --kind SpeechServices --sku F0 --location northcentralus --yes
az cognitiveservices account create --name cs345-language --resource-group cs345-speech-rg \
  --kind TextAnalytics --sku F0 --location northcentralus --yes
az monitor log-analytics workspace create --resource-group cs345-speech-rg \
  --workspace-name cs345-logs --location northcentralus
az monitor app-insights component create --app cs345-insights --location northcentralus \
  --resource-group cs345-speech-rg \
  --workspace "$(az monitor log-analytics workspace show -g cs345-speech-rg -n cs345-logs --query id -o tsv)"
```

Keys and the connection string go in `.env` locally (see `.env.example`) and in App Service Application Settings when deployed (`scripts/set_app_settings.sh`).

## Deploy (Part F)

```bash
az appservice plan create --name cs345-speech-plan --resource-group cs345-speech-rg \
  --sku F1 --is-linux --location northcentralus
az webapp create --resource-group cs345-speech-rg --plan cs345-speech-plan \
  --name <APP_NAME> --runtime "PYTHON:3.12"
az webapp config set --resource-group cs345-speech-rg --name <APP_NAME> --startup-file "startup.sh"
scripts/set_app_settings.sh <APP_NAME>
az webapp up --resource-group cs345-speech-rg --name <APP_NAME> --runtime "PYTHON:3.12" --sku F1
```

`startup.sh` installs ffmpeg (not included in the App Service Python image) and starts gunicorn.

## Observability (Part G)

`telemetry.py` initializes Azure Monitor before Flask or any Azure SDK is imported. Per call it emits:

- Spans under `pipeline.process`: `stage.speech_to_text`, `stage.language_analysis` (with nested `stage.pii_redaction`), `stage.text_to_speech`.
- Metrics: `stt_confidence`, `stt_duration_seconds`, `stt_word_count`, `language_entity_count`, `language_keyphrase_count`, `language_sentiment`, `tts_char_count`, `stage_stt_ms`, `stage_language_ms`, `stage_tts_ms`, plus product metrics `call_flagged` (domain metric, histogram with a `flag_reason` attribute so its average is the share flagged), `pii_redacted_count`, `stage_pii_ms` and `language_text_chars`.
- A `pipeline_completed` or `pipeline_error` event as attributes on the request span.
- Alert rule `low-stt-confidence` (average `stt_confidence` below 0.8 over 5 minutes).

KQL used for the report is in `queries.kql`.

## Part I extensions

**Track 1: escalation flagging and PII redaction.** Rules above; PII categories redacted: person, phone number, email, address, credit card number. "manager" is not redacted because only these categories are requested.

**Track 2: cost model.** `cost_model.py` projects the monthly cost of 1,000 calls per day after free tiers, from per-call averages measured with KQL (`stt_duration_seconds`, `tts_char_count`, `language_text_chars`) and unit prices from the Azure pricing calculator. The deployed app itself uses only free tiers.

## Tests

Five scripted recordings with invented customers (555-01xx phones, example.com emails): (1) calm shipping question, not flagged; (2) angry refund and manager demand, flagged; (3) billing question with phone and email read aloud, not flagged, PII redacted; (4) frustrated then happy, not flagged; (5) home address and card last four, then threat to cancel, flagged, PII redacted.

## Known limits

- Synchronous Language calls reject documents over about 5,120 characters (roughly 5 to 6 minutes of speech). Longer calls would need chunking.
- `language` is always `en-US`; no language detection is performed.
- Speech F0 allows one concurrent recognition, and App Service F1 has 60 CPU-minutes per day.
