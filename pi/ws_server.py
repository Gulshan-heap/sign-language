"""Raspberry Pi edition: the Pi does all the recognition, any browser on the Wi-Fi watches.

    python pi/ws_server.py --camera 0        # USB webcam on the Pi      then open http://<pi-ip>:8765
    python pi/ws_server.py --camera picam    # Pi Camera Module (ribbon)
    python pi/ws_server.py --camera browser  # the laptop's / phone's own camera, frames sent to the Pi over WebSocket

Pipeline (one process, three parts):
    camera source   USB webcam / Pi camera thread, or JPEGs arriving from the browser -> newest frame only
                    (old frames are dropped, so lag never builds up)
    worker thread   frame -> MediaPipe (pose + hands) -> WordSegmenter -> word model -> event
    asyncio loop    WebSocket server: pushes JSON (landmarks, state, words) [+ JPEG preview] to every browser,
                    receives commands {"cmd": "lang"|"thr"|"reset"} and, in browser mode, binary JPEG frames.
                    GET / serves pi/index.html.

Browser-camera mode needs a "secure context": browsers only allow the camera on https:// or http://localhost.
Either tunnel the port over SSH from the laptop (ssh -L 8765:localhost:8765 pi@signpi.local, then open
http://localhost:8765), or start the server with --cert / --key (self-signed certificate) and open https://<pi-ip>:8765.

Why WebSocket and not one HTTP POST per frame (web/live.py)? One open connection carries frames up and results down
without per-frame HTTP headers, and when the camera is on the Pi the server can *push* results to many viewers.

Optional hardware (gpiozero, BCM numbering): LED on GPIO17 = "signing", LED on GPIO27 = "word accepted",
active buzzer on GPIO22 = short beep on each accepted word. Missing gpiozero (e.g. on a laptop) just disables them.
"""
import argparse
import asyncio
import json
import sys
import threading
import time
from http import HTTPStatus
from pathlib import Path

import cv2
import numpy as np
from websockets.asyncio.server import broadcast, serve
from websockets.datastructures import Headers
from websockets.http11 import Response

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slr.inference import Models  # noqa: E402
from slr.landmarks import LH, N_POSE, RH, FrameTracker  # noqa: E402
from slr.live import WordSegmenter  # noqa: E402

PAGE = Path(__file__).with_name("index.html")


class Gpio:
    """Status LEDs + buzzer. Every method is a no-op when gpiozero / the GPIO header is not available."""

    def __init__(self, enabled):
        self.ok = False
        if not enabled:
            return
        try:
            from gpiozero import LED, Buzzer
            self.busy, self.good, self.buzz = LED(17), LED(27), Buzzer(22)
            self.ok = True
            print("GPIO: LEDs on 17 / 27, buzzer on 22")
        except Exception as e:                       # not a Pi, or pins in use
            print("GPIO disabled:", e)

    def signing(self, on):
        if self.ok:
            self.busy.value = on

    def accepted(self):
        if self.ok:
            self.good.blink(on_time=0.6, n=1)
            self.buzz.beep(on_time=0.08, n=1)


class Camera:
    """Reads frames in its own thread and keeps only the newest one."""

    def __init__(self, src, width, height):
        self.frame, self.n, self.lock = None, 0, threading.Lock()
        if src == "picam":                           # ribbon-cable Pi Camera Module (libcamera; OpenCV cannot open it)
            from picamera2 import Picamera2
            self.picam = Picamera2()
            self.picam.configure(self.picam.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"}))   # "RGB888" arrives in BGR order, like OpenCV
            self.picam.start()
            threading.Thread(target=self._run_picam, daemon=True).start()
            return
        self.cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            sys.exit(f"cannot open camera {src!r} (try --camera 1, or check `ls /dev/video*`)")
        self.loop_file = not str(src).isdigit()      # a video file: replay it forever (handy for testing)
        threading.Thread(target=self._run, daemon=True).start()

    def _run_picam(self):
        while True:
            img = self.picam.capture_array()
            with self.lock:
                self.frame, self.n = img, self.n + 1

    def _run(self):
        while True:
            ok, img = self.cap.read()
            if not ok:
                if self.loop_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.05)
                continue
            with self.lock:
                self.frame, self.n = img, self.n + 1
            if self.loop_file:
                time.sleep(1 / 15)                     # play files at ~15 fps like a camera

    def latest(self, after):
        """-> (frame number, image) newer than `after`, or (after, None)."""
        with self.lock:
            return (self.n, self.frame) if self.n > after else (after, None)


class BrowserCamera:
    """Frames come from the browser as JPEG bytes over the WebSocket. Same latest() interface as Camera.

    push() runs on the asyncio loop and only stores the bytes; decoding happens in the worker thread.
    """

    def __init__(self):
        self.jpg, self.n, self.lock = None, 0, threading.Lock()

    def push(self, data):
        with self.lock:
            self.jpg, self.n = data, self.n + 1

    def latest(self, after):
        with self.lock:
            if self.n <= after:
                return after, None
            n, data = self.n, self.jpg
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return n, img                                  # img is None for a bad JPEG; the worker skips it


def pts(a):
    return [None if not np.isfinite(p).all() else [round(float(p[0]), 3), round(float(p[1]), 3)] for p in a]


class Engine:
    """Runs in a worker thread: newest frame -> keypoints -> segmenter -> messages for the browsers."""

    def __init__(self, args, publish):
        self.models = Models(args.models)
        if not self.models.words:
            sys.exit(f"no word models in {args.models}")
        self.lang = args.lang if args.lang in self.models.words else next(iter(self.models.words))
        self.thr, self.publish, self.gpio = args.thr, publish, Gpio(not args.no_gpio)
        self.browser = args.camera == "browser"
        self.cam = BrowserCamera() if self.browser else Camera(args.camera, args.width, args.height)
        self.tracker = FrameTracker()
        self.commands, self.lock = [], threading.Lock()
        self.seg = self._new_segmenter()
        print("models:", {l: len(self.models.words[l][1]) for l in self.models.words}, "| using", self.lang)

    def _new_segmenter(self):
        lang = self.lang
        return WordSegmenter(lambda seq: self.models.word(lang, seq), threshold=self.thr)

    def command(self, msg):
        """Called from the WebSocket side. Settings change at once; the segmenter is rebuilt by the worker thread."""
        with self.lock:
            if msg.get("cmd") == "lang" and msg.get("value") in self.models.words:
                self.lang = msg["value"]
                self.commands.append("new")
            elif msg.get("cmd") == "thr":
                self.thr = min(0.95, max(0.05, float(msg.get("value", 0.5))))
                self.commands.append("thr")
            elif msg.get("cmd") == "reset":
                self.commands.append("reset")

    def _apply_commands(self):
        with self.lock:
            cmds, self.commands = self.commands, []
        if "new" in cmds:
            self.seg = self._new_segmenter()
        elif "reset" in cmds:
            self.seg.reset()
        self.seg.threshold = self.thr

    def info(self):
        return {"type": "info", "langs": {l: self.models.words[l][1] for l in self.models.words},
                "lang": self.lang, "thr": self.thr, "source": "browser" if self.browser else "pi"}

    def run(self):
        last, fps_t, fps_n, fps = 0, time.time(), 0, 0.0
        while True:
            n, img = self.cam.latest(last)
            if img is None:
                last = n                               # skip a bad frame instead of retrying it
                time.sleep(0.005)
                continue
            last = n
            self._apply_commands()
            t0 = time.time()
            fr = self.tracker(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))   # un-mirrored, as in training
            ev = self.seg.push(fr)
            ms = round((time.time() - t0) * 1000)
            self.gpio.signing(self.seg.state == "signing")
            fps_n += 1
            if time.time() - fps_t >= 1:
                fps, fps_n, fps_t = fps_n / (time.time() - fps_t), 0, time.time()
            msg = {"type": "frame", "state": self.seg.state, "ms": ms, "fps": round(fps, 1), "lang": self.lang,
                   "pose": pts(fr[:N_POSE]), "hands": [h for h in (pts(fr[LH]), pts(fr[RH])) if h[0]]}
            if ev:
                ev.pop("seq", None)
                msg["event"] = ev
                if ev["accepted"]:
                    self.gpio.accepted()
                print("sign:", ev["top"][0] if ev["top"] else ev.get("reason"), "accepted" if ev["accepted"] else "")
            jpg = None
            if not self.browser:                       # the browser already shows its own camera
                ok, enc = cv2.imencode(".jpg", cv2.resize(img, (320, 240)), [cv2.IMWRITE_JPEG_QUALITY, 60])
                jpg = enc.tobytes() if ok else None
            self.publish(msg, jpg)


async def main_async(args):
    clients = set()
    loop = asyncio.get_running_loop()

    def publish(msg, jpg):                      # called from the worker thread
        def send():
            broadcast(clients, json.dumps(msg))
            if jpg is not None and args.preview:
                broadcast(clients, jpg)
        loop.call_soon_threadsafe(send)

    engine = Engine(args, publish)

    async def handler(ws):
        clients.add(ws)
        print("browser connected:", ws.remote_address, "| clients:", len(clients))
        try:
            await ws.send(json.dumps(engine.info()))
            async for raw in ws:
                if isinstance(raw, bytes):             # a camera frame (JPEG) from the browser
                    if engine.browser:
                        engine.cam.push(raw)
                    continue
                try:
                    engine.command(json.loads(raw))
                except (ValueError, TypeError, AttributeError):
                    continue
                await ws.send(json.dumps(engine.info()))
        finally:
            clients.discard(ws)

    def http(connection, request):
        """Plain GET / (not a WebSocket upgrade) -> the dashboard page."""
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None
        if request.path not in ("/", "/index.html"):
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
        body = PAGE.read_bytes()
        return Response(200, "OK", Headers([("Content-Type", "text/html; charset=utf-8"),
                                            ("Content-Length", str(len(body)))]), body)

    tls = None
    if args.cert:
        import ssl
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(args.cert, args.key)
    scheme = "https" if tls else "http"

    threading.Thread(target=engine.run, daemon=True).start()
    async with serve(handler, args.host, args.port, process_request=http, ssl=tls, max_size=2 ** 21):
        print(f"open {scheme}://<this-pi's-ip>:{args.port}  (WebSocket on the same port, path /ws)")
        if engine.browser and not tls:
            print(f"browser camera: the camera only works on https or localhost. From the laptop run\n"
                  f"  ssh -L {args.port}:localhost:{args.port} pi@<pi-ip>   and open http://localhost:{args.port}\n"
                  f"or restart with --cert cert.pem --key key.pem")
        await asyncio.get_running_loop().create_future()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--lang", default="asl", choices=["asl", "isl"])
    ap.add_argument("--camera", default="browser",
                    help="USB camera index (0 = /dev/video0), 'picam' for the Pi Camera Module, "
                         "'browser' for the viewer's laptop/phone camera, or a video file")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--thr", type=float, default=0.5, help="min. confidence to accept a word")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-preview", dest="preview", action="store_false", help="do not stream JPEG preview frames")
    ap.add_argument("--no-gpio", action="store_true")
    ap.add_argument("--cert", help="TLS certificate (PEM) to serve https/wss; needed for the browser camera off localhost")
    ap.add_argument("--key", help="TLS private key (PEM) for --cert")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
