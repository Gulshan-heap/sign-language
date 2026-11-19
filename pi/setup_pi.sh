#!/usr/bin/env bash
# One-time setup on Raspberry Pi OS (64-bit, Bookworm). Run from the project folder:  bash pi/setup_pi.sh
# The camera is the laptop's own webcam: the laptop's browser sends frames to the Pi over a WebSocket.
set -e
cd "$(dirname "$0")/.."

[ "$(uname -m)" = "aarch64" ] || { echo "Needs 64-bit Raspberry Pi OS (uname -m = aarch64). MediaPipe has no 32-bit wheels."; exit 1; }

sudo apt update
sudo apt install -y python3-venv python3-gpiozero python3-lgpio libgl1 openssl

# --system-site-packages lets the venv see gpiozero/lgpio, which come from apt (not pip)
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt websockets "numpy<2"
# numpy<2: mediapipe 0.10.14 is built against numpy 1.x

# fetch MediaPipe's hand + pose models now (~14 MB), so the Pi can later run without Internet
python -c "from slr.landmarks import model_path; print(model_path('hand_landmarker.task')); print(model_path('pose_landmarker_lite.task'))"

# self-signed HTTPS certificate: browsers only allow the webcam on https:// (or localhost)
if [ ! -f pi/cert.pem ]; then
  openssl req -x509 -newkey rsa:2048 -nodes -days 825 -keyout pi/key.pem -out pi/cert.pem \
    -subj "/CN=$(hostname).local" -addext "subjectAltName=DNS:$(hostname).local,IP:$(hostname -I | awk '{print $1}')"
fi

IP=$(hostname -I | awk '{print $1}')
echo
echo "Run:   . .venv/bin/activate && python pi/ws_server.py --cert pi/cert.pem --key pi/key.pem"
echo "Then on the laptop open  https://$(hostname).local:8765  (or https://$IP:8765),"
echo "accept the certificate warning once (Advanced -> Proceed), press Start camera and allow the camera."
