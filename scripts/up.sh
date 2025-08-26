#!/bin/bash
set -e

cd "$(dirname "$0")/.."


uvicorn web_app.main:app --reload --host 0.0.0.0 --port 8000