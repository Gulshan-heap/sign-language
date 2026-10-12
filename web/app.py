"""Web interface: record a sign with the laptop camera in the browser, get the prediction, log it, see accuracy.

  python web/app.py --models models          # then open http://localhost:5000

Words:   Record -> the clip is uploaded, keypoints are extracted on the server, the word model answers (top-3).
Letters: live prediction from the camera; pick the true letter and press "Log" to add it to the evaluation.
If you pick the TRUE word/letter before recording, the result is logged to results/web_log.csv and the
Evaluation panel shows accuracy, per-class accuracy and the most frequent confusions. Clips are also saved
to clips/<lang>/<word>/ so tools/evaluate_clips.py can re-run them later.
Everything runs locally; nothing is uploaded anywhere. (Camera access works on http://localhost.)
"""
import argparse
import csv
import tempfile
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from slr.inference import Models  # noqa: E402
from slr.landmarks import FrameTracker, HandDetector, LH, RH  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
LOCK = threading.Lock()            # MediaPipe objects are not thread-safe
STATE = {}
LOG_FIELDS = ["time", "kind", "lang", "truth", "top1", "top2", "top3", "prob1", "correct", "clip"]


def log_row(row):
    path = STATE["results"] / "web_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def read_log():
    path = STATE["results"] / "web_log.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@app.get("/")
def index():
    return INDEX_HTML


@app.get("/api/info")
def info():
    M = STATE["models"]
    return jsonify({
        "words": {l: M.words[l][1] for l in M.words},
        "letters": {l: M.letters[l].classes if not isinstance(M.letters[l], dict) else M.letters[l]["classes"] for l in M.letters},
    })


@app.post("/api/word")
def word():
    M, lang, truth = STATE["models"], request.form["lang"], request.form.get("truth", "").strip()
    if lang not in M.words:
        return jsonify(error=f"No {lang.upper()} word model in {STATE['models_dir']} (expected {lang}_words.pt)"), 400
    f = request.files["video"]
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / ("clip" + (Path(f.filename or "").suffix or ".webm"))
        f.save(src)
        cap, kp = cv2.VideoCapture(str(src)), []
        t0 = time.time()
        with LOCK:
            tracker = FrameTracker()
            i = 0
            while True:
                ok, img = cap.read()
                if not ok or len(kp) >= 300:
                    break
                if i % 2 == 0:
                    if max(img.shape[:2]) > 960:
                        s = 960 / max(img.shape[:2]); img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                    kp.append(tracker(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                i += 1
            tracker.close()
        cap.release()
        if len(kp) < 8:
            return jsonify(error="Clip too short or could not be decoded - record at least 1-2 seconds."), 400
        seq = np.stack(kp)
        hands_pct = float((~np.isnan(seq[:, LH.start:RH.stop, 0]).all(1)).mean())
        top = [{"label": l, "prob": p} for l, p in M.word(lang, seq)]
        clip = ""
        if truth:
            d = ROOT / "clips" / lang / truth
            d.mkdir(parents=True, exist_ok=True)
            dst = d / f"web_{int(time.time())}{src.suffix}"
            dst.write_bytes(src.read_bytes()); clip = str(dst.relative_to(ROOT))
    if truth:
        t = [x["label"] for x in top] + ["", ""]
        log_row({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "word", "lang": lang, "truth": truth,
                 "top1": t[0], "top2": t[1], "top3": t[2], "prob1": f"{top[0]['prob']:.3f}",
                 "correct": int(t[0] == truth), "clip": clip})
    return jsonify(top=top, frames=len(kp), hands_pct=hands_pct, seconds=round(time.time() - t0, 1), logged=bool(truth))


@app.post("/api/letter")
def letter():
    M, lang, truth = STATE["models"], request.form["lang"], request.form.get("truth", "").strip()
    if lang not in M.letters:
        return jsonify(error=f"No {lang.upper()} letters model in {STATE['models_dir']} ({lang}_letters_np.npz)"), 400
    img = cv2.imdecode(np.frombuffer(request.files["image"].read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="bad image"), 400
    with LOCK:
        det = STATE.setdefault(f"hands_{lang}", HandDetector(1 if lang == "asl" else 2, video=False))
        found = det(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pred = M.letter(lang, found) if found else None
    if truth and request.form.get("log"):
        log_row({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "letter", "lang": lang, "truth": truth,
                 "top1": pred[0] if pred else "(no hand)", "top2": "", "top3": "",
                 "prob1": f"{pred[1]:.3f}" if pred else "0", "correct": int(bool(pred) and pred[0] == truth), "clip": ""})
    return jsonify(letter=pred[0] if pred else None, prob=pred[1] if pred else 0.0, hands=len(found))


@app.get("/api/stats")
def stats():
    rows = [r for r in read_log() if r["kind"] == request.args.get("kind", "word") and r["lang"] == request.args.get("lang", "isl")]
    if not rows:
        return jsonify(n=0)
    per, conf = defaultdict(list), Counter()
    for r in rows:
        per[r["truth"]].append(int(r["correct"]))
        if not int(r["correct"]):
            conf[(r["truth"], r["top1"])] += 1
    top3 = [r["truth"] in (r["top1"], r["top2"], r["top3"]) for r in rows]
    return jsonify(n=len(rows), top1=float(np.mean([int(r["correct"]) for r in rows])), top3=float(np.mean(top3)),
                   per_class=[{"label": k, "acc": float(np.mean(v)), "n": len(v)} for k, v in sorted(per.items())],
                   confusions=[{"truth": a, "pred": b, "count": c} for (a, b), c in conf.most_common(8)])


@app.get("/api/log.csv")
def log_csv():
    path = STATE["results"] / "web_log.csv"
    if not path.exists():
        return "no log yet", 404
    return send_file(path, as_attachment=True)


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sign Language Evaluator</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1c2230;--mut:#667085;--line:#e3e6ec;--acc:#2f6fed;--ok:#1a9d5b;--bad:#d64545}
@media (prefers-color-scheme:dark){:root{--bg:#12151c;--card:#1b2029;--ink:#e8ebf2;--mut:#98a2b3;--line:#2b3240;--acc:#6c9bff;--ok:#38c585;--bad:#ff7a7a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,Segoe UI,sans-serif}
header{padding:16px 20px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:18px;margin:0;flex:1}main{max-width:1100px;margin:0 auto;padding:16px;display:grid;gap:16px;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr)}
@media (max-width:820px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.card h2{font-size:14px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);margin:0 0 12px}
video{width:100%;border-radius:10px;background:#000;transform:scaleX(-1);aspect-ratio:4/3;object-fit:cover}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0}
button,select,input{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px}
button{cursor:pointer}button.primary{background:var(--acc);border-color:var(--acc);color:#fff}button:disabled{opacity:.5;cursor:default}
button.rec{background:var(--bad);border-color:var(--bad);color:#fff}
.tabs button.on{background:var(--acc);border-color:var(--acc);color:#fff}
.bar{height:22px;border-radius:6px;background:var(--line);position:relative;margin:6px 0;overflow:hidden}
.bar>i{position:absolute;inset:0 auto 0 0;background:var(--acc);opacity:.85}.bar>span{position:relative;padding:0 8px;line-height:22px;font-size:13px;color:#fff;mix-blend-mode:normal;text-shadow:0 0 3px #0008}
.big{font-size:40px;font-weight:700;margin:4px 0}.mut{color:var(--mut)}.ok{color:var(--ok)}.bad{color:var(--bad)}
table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:5px 6px;border-bottom:1px solid var(--line)}
#count{font-size:64px;font-weight:700;position:absolute;inset:0;display:none;align-items:center;justify-content:center;color:#fff;text-shadow:0 0 12px #000}
.wrap{position:relative}
</style></head><body>
<header><h1>Sign Language Evaluator</h1>
<div class="tabs row" style="margin:0"><button id="tw" class="on">Words</button><button id="tl">Letters</button></div>
<select id="lang"><option value="isl">ISL</option><option value="asl">ASL</option></select></header>
<main>
<section class="card"><h2>Camera</h2>
<div class="wrap"><video id="cam" autoplay muted playsinline></video><div id="count"></div></div>
<div id="wordCtl">
 <div class="row"><label>True word (optional, enables evaluation)</label>
  <select id="truthW" style="min-width:160px"></select></div>
 <div class="row"><label>Length <input id="secs" type="number" min="1" max="8" value="3" style="width:64px"> s</label>
  <button id="rec" class="primary">&#9679; Record</button>
  <label class="mut">or <input id="file" type="file" accept="video/*"></label></div>
 <p class="mut" style="margin:6px 0 0">Press Record, get ready during the 3-2-1, sign once, hold still at the end.
 Keep both hands and your upper body in view, with good light.</p>
</div>
<div id="letterCtl" style="display:none">
 <div class="row"><label>True letter</label><select id="truthL"></select>
  <button id="logL" class="primary">Log this frame</button><label><input type="checkbox" id="live" checked> live</label></div>
</div></section>
<section class="card"><h2 id="resTitle">Result</h2><div id="result" class="mut">Nothing yet.</div></section>
<section class="card" style="grid-column:1/-1"><h2>Evaluation (from logged attempts)</h2><div id="stats" class="mut">Pick the true word, record, repeat. Accuracy appears here.</div>
<div class="row"><button id="refresh">Refresh</button><a href="/api/log.csv" class="mut">download log (csv)</a></div></section>
</main>
<script>
const $=id=>document.getElementById(id);let mode="words",info={words:{},letters:{}},stream,busy=false;
async function boot(){
  try{stream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480},audio:false});$("cam").srcObject=stream}
  catch(e){$("result").innerHTML='<span class="bad">Camera blocked: '+e.message+'. Allow camera access for localhost and reload.</span>'}
  info=await (await fetch("/api/info")).json();fill();loop()}
function fill(){const l=$("lang").value;
  $("truthW").innerHTML='<option value="">— unknown (just predict) —</option>'+(info.words[l]||[]).map(w=>`<option>${w}</option>`).join("");
  $("truthL").innerHTML='<option value="">—</option>'+(info.letters[l]||[]).map(w=>`<option>${w}</option>`).join("");
  if(!(info.words[l]||[]).length&&mode=="words")$("result").innerHTML='<span class="bad">No '+l.toUpperCase()+' word model loaded (models/'+l+'_words.pt).</span>';
  stats()}
$("lang").onchange=fill;
function setMode(m){mode=m;$("tw").classList.toggle("on",m=="words");$("tl").classList.toggle("on",m=="letters");
  $("wordCtl").style.display=m=="words"?"":"none";$("letterCtl").style.display=m=="letters"?"":"none";$("result").textContent="Nothing yet.";stats()}
$("tw").onclick=()=>setMode("words");$("tl").onclick=()=>setMode("letters");
function bars(top,truth){return top.map((t,i)=>`<div class="bar"><i style="width:${(t.prob*100).toFixed(0)}%"></i><span>${i+1}. ${t.label} &nbsp;${(t.prob*100).toFixed(0)}%</span></div>`).join("")}
async function sendWord(blob,name){
  $("result").innerHTML='<span class="mut">Analysing… (a few seconds)</span>';
  const fd=new FormData();fd.append("video",blob,name);fd.append("lang",$("lang").value);fd.append("truth",$("truthW").value);
  const r=await fetch("/api/word",{method:"POST",body:fd});const j=await r.json();
  if(j.error){$("result").innerHTML='<span class="bad">'+j.error+'</span>';return}
  const t=$("truthW").value;let verdict="";
  if(t)verdict=j.top[0].label==t?'<p class="ok"><b>Correct</b> — logged.</p>':'<p class="bad"><b>Wrong</b> (expected '+t+') — logged.</p>';
  const warn=j.hands_pct<.3?'<p class="bad">Hands were visible in only '+(j.hands_pct*100).toFixed(0)+'% of frames — re-record with hands in view.</p>':"";
  $("result").innerHTML=`<div class="big">${j.top[0].label}</div>${verdict}${warn}${bars(j.top)}<p class="mut">${j.frames} frames, hands seen in ${(j.hands_pct*100).toFixed(0)}%, ${j.seconds}s</p>`;
  stats()}
$("rec").onclick=async()=>{
  if(!stream||busy)return;busy=true;$("rec").disabled=true;
  for(let n=3;n>0;n--){$("count").style.display="flex";$("count").textContent=n;await new Promise(r=>setTimeout(r,1000))}
  $("count").style.display="none";$("rec").textContent="● Recording…";$("rec").classList.add("rec");
  const mr=new MediaRecorder(stream,{mimeType:MediaRecorder.isTypeSupported("video/webm;codecs=vp8")?"video/webm;codecs=vp8":""});const ch=[];
  mr.ondataavailable=e=>ch.push(e.data);const done=new Promise(r=>mr.onstop=r);
  mr.start();await new Promise(r=>setTimeout(r,Math.min(8,Math.max(1,+$("secs").value))*1000));mr.stop();await done;
  $("rec").textContent="● Record";$("rec").classList.remove("rec");
  await sendWord(new Blob(ch,{type:"video/webm"}),"clip.webm");$("rec").disabled=false;busy=false};
$("file").onchange=async e=>{const f=e.target.files[0];if(f)await sendWord(f,f.name);e.target.value=""};
function frameBlob(){const c=document.createElement("canvas"),v=$("cam");c.width=v.videoWidth||640;c.height=v.videoHeight||480;
  c.getContext("2d").drawImage(v,0,0,c.width,c.height);return new Promise(r=>c.toBlob(r,"image/jpeg",.9))}
async function sendLetter(log){const fd=new FormData();fd.append("image",await frameBlob(),"f.jpg");fd.append("lang",$("lang").value);
  fd.append("truth",$("truthL").value);if(log)fd.append("log","1");
  const j=await (await fetch("/api/letter",{method:"POST",body:fd})).json();
  if(j.error){$("result").innerHTML='<span class="bad">'+j.error+'</span>';return}
  $("result").innerHTML=j.letter?`<div class="big">${j.letter}</div><div class="mut">${(j.prob*100).toFixed(0)}% confidence</div>`:'<span class="mut">No hand detected</span>';
  if(log){stats()}}
async function loop(){while(true){if(mode=="letters"&&$("live").checked&&stream&&!busy){busy=true;try{await sendLetter(false)}catch(e){}busy=false}
  await new Promise(r=>setTimeout(r,450))}}
$("logL").onclick=async()=>{if(!$("truthL").value){alert("Pick the true letter first");return}await sendLetter(true)};
async function stats(){const j=await (await fetch(`/api/stats?kind=${mode=="words"?"word":"letter"}&lang=${$("lang").value}`)).json();
  if(!j.n){$("stats").innerHTML='<span class="mut">No logged attempts yet for this language/mode.</span>';return}
  $("stats").innerHTML=`<p><b>${j.n}</b> attempts &nbsp; top-1 <b>${(j.top1*100).toFixed(0)}%</b> &nbsp; top-3 <b>${(j.top3*100).toFixed(0)}%</b></p>
  <table><tr><th>Class</th><th>Accuracy</th><th>Attempts</th></tr>${j.per_class.map(p=>`<tr><td>${p.label}</td><td>${(p.acc*100).toFixed(0)}%</td><td>${p.n}</td></tr>`).join("")}</table>
  ${j.confusions.length?'<p class="mut">Most common mistakes: '+j.confusions.map(c=>`${c.truth}→${c.pred} (${c.count})`).join(", ")+'</p>':""}`}
$("refresh").onclick=stats;boot();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open it from another device on your network")
    a = ap.parse_args()
    STATE.update(models=Models(a.models), models_dir=a.models, results=Path(a.results))
    M = STATE["models"]
    print("word models:", list(M.words), "| letter models:", list(M.letters))
    print(f"open http://localhost:{a.port}")
    app.run(host=a.host, port=a.port, threaded=True)


if __name__ == "__main__":
    main()
