#!/bin/bash
# App Service startup: install ffmpeg (not in the Python image), then run gunicorn.
# One worker keeps the in-memory session log in one process; Speech F0 allows
# only one concurrent recognition anyway.
apt-get update && apt-get install -y ffmpeg
gunicorn --bind=0.0.0.0:${PORT:-8000} --timeout 600 --workers 1 --threads 4 app:app
