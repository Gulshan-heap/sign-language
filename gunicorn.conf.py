# Read automatically by gunicorn from the project root.
import os

bind = "0.0.0.0:" + os.environ.get("PORT", "10000")
workers = 1          # one process: the MediaPipe/torch models are loaded once and the live session state is shared
threads = 4
timeout = 180        # the first start downloads MediaPipe models and loads torch
