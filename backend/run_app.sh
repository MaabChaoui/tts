#!/bin/sh
# Start the one-word replacement tool at http://127.0.0.1:7861 (local only). Stop with Ctrl+C.
cd "$(dirname "$0")" && exec .venv-f5/bin/python app.py
