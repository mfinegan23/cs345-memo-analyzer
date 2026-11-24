#!/bin/bash
# Copies your local .env values into App Service Application Settings without
# printing them. Usage (from project root):  scripts/set_app_settings.sh <APP_NAME>
set -euo pipefail
APP="${1:?usage: scripts/set_app_settings.sh <APP_NAME>}"
RG="cs345-speech-rg"
[ -f .env ] || { echo ".env not found. Run from the project root."; exit 1; }
set -a; source .env; set +a
[ -f .internal_token ] || openssl rand -hex 16 > .internal_token   # git-ignored
TOKEN="$(cat .internal_token)"
# --output none: this command otherwise prints every setting value in plain text
az webapp config appsettings set -g "$RG" -n "$APP" --output none --settings \
  AZURE_SPEECH_KEY="$AZURE_SPEECH_KEY" \
  AZURE_SPEECH_REGION="$AZURE_SPEECH_REGION" \
  AZURE_LANGUAGE_KEY="$AZURE_LANGUAGE_KEY" \
  AZURE_LANGUAGE_ENDPOINT="$AZURE_LANGUAGE_ENDPOINT" \
  APPLICATIONINSIGHTS_CONNECTION_STRING="$APPLICATIONINSIGHTS_CONNECTION_STRING" \
  INTERNAL_TOKEN="$TOKEN" \
  SCM_DO_BUILD_DURING_DEPLOYMENT=true
echo "Settings applied (values not printed). Names now set:"
az webapp config appsettings list -g "$RG" -n "$APP" --query "[].name" -o tsv
