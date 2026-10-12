"""Entry point for hosts that run `gunicorn app:app` (Render's default). Serves the live dashboard in web/live.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "web"))
from live import app  # noqa: E402,F401
