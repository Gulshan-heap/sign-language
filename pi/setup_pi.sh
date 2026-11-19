#!/usr/bin/env bash
# One-time setup on Raspberry Pi OS (64-bit, Bookworm). Run from the project folder:  bash pi/setup_pi.sh
set -e
cd "$(dirname "$0")/.."

[ "$(uname -m)" = "aarch64" ] || { echo "Needs 64-bit Raspberry Pi OS (uname -m = aarch64). MediaPipe has no 32-bit wheels."; exit 1; }

sudo apt update
sudo apt install -y python3-venv python3-picamera2 python3-gpiozero python3-lgpio libgl1 v4l-utils

# --system-site-packages lets the venv see picamera2 and gpiozero/lgpio, which come from apt (not pip)
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt websockets "numpy<2"
# numpy<2: apt's picamera2 is built against numpy 1.x; a numpy 2 in the venv breaks `import picamera2`
# requirements.txt pulls opencv-python-headless: no desktop GUI needed on the Pi

# fetch MediaPipe's hand + pose models now (~14 MB), so the Pi can later run without Internet
python -c "from slr.landmarks import model_path; print(model_path('hand_landmarker.task')); print(model_path('pose_landmarker_lite.task'))"

echo
echo "Cameras found:"; v4l2-ctl --list-devices || true
echo
echo "Test:  . .venv/bin/activate && python pi/ws_server.py --camera 0        (USB webcam)"
echo "       . .venv/bin/activate && python pi/ws_server.py --camera picam    (Pi Camera Module)"
echo "Then open http://$(hostname -I | awk '{print $1}'):8765 on a phone or laptop on the same Wi-Fi."
