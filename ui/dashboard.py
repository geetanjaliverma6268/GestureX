"""Optional web dashboard (FastAPI) for monitoring the running system.

This never replaces the real-time OpenCV window - it is a read-only monitoring
view. ``main`` pushes a status snapshot into :class:`DashboardState` each frame;
a browser polls ``/api/status`` and renders mode, lock state, per-hand gesture,
confidence, last action, FPS, hand count and the gesture/action history.
"""

import threading
import time

from utils.logger import get_logger

log = get_logger("dashboard")


class DashboardState:
    """Thread-safe latest-status snapshot shared between the CV loop and web."""

    def __init__(self, history_len=30):
        self._lock = threading.Lock()
        self._history_len = history_len
        self._state = {
            "running": False,
            "mode": "NORMAL",
            "locked": True,
            "control_on": True,
            "fps": 0.0,
            "hands": 0,
            "left": {"gesture": "NONE", "confidence": 0.0},
            "right": {"gesture": "NONE", "confidence": 0.0},
            "last_action": "-",
            "gesture_history": [],
            "action_history": [],
        }

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                if k in self._state:
                    self._state[k] = v

    def push_event(self, gesture, action):
        entry = {"time": time.strftime("%H:%M:%S"), "gesture": gesture, "action": action}
        with self._lock:
            self._state["gesture_history"].insert(0, entry)
            self._state["gesture_history"] = self._state["gesture_history"][:self._history_len]
            self._state["action_history"] = self._state["gesture_history"]

    def snapshot(self):
        with self._lock:
            s = dict(self._state)
            s["gesture_history"] = list(self._state["gesture_history"])
            s["action_history"] = list(self._state["action_history"])
            return s


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Touchless Control - Dashboard</title>
<style>
 :root{color-scheme:dark} body{font-family:system-ui,Segoe UI,Arial;margin:0;background:#0f1220;color:#e8eaf2}
 header{padding:18px 24px;background:#171a2b;border-bottom:1px solid #2a2f4a}
 h1{font-size:18px;margin:0;letter-spacing:.5px} .wrap{padding:20px 24px;display:grid;gap:16px;
 grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}
 .card{background:#171a2b;border:1px solid #2a2f4a;border-radius:12px;padding:16px}
 .k{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:#9aa3c7}
 .v{font-size:22px;margin-top:6px;font-weight:600} .pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:13px}
 .on{background:#12351f;color:#5ef08a}.off{background:#3a1414;color:#ff8b8b}
 table{width:100%;border-collapse:collapse;font-size:13px} td,th{padding:6px 8px;border-bottom:1px solid #262b45;text-align:left}
 .full{grid-column:1/-1}
</style></head><body>
<header><h1>AI TOUCHLESS COMPUTER CONTROL &mdash; Dashboard</h1></header>
<div class="wrap">
 <div class="card"><div class="k">Mode</div><div class="v" id="mode">-</div></div>
 <div class="card"><div class="k">Control Lock</div><div class="v"><span id="lock" class="pill off">-</span></div></div>
 <div class="card"><div class="k">System</div><div class="v"><span id="control" class="pill off">-</span></div></div>
 <div class="card"><div class="k">FPS</div><div class="v" id="fps">-</div></div>
 <div class="card"><div class="k">Hands Detected</div><div class="v" id="hands">-</div></div>
 <div class="card"><div class="k">Left Hand</div><div class="v" id="left">-</div><div class="k" id="leftc"></div></div>
 <div class="card"><div class="k">Right Hand</div><div class="v" id="right">-</div><div class="k" id="rightc"></div></div>
 <div class="card"><div class="k">Last Action</div><div class="v" id="action">-</div></div>
 <div class="card full"><div class="k">Gesture / Action History</div>
  <table><thead><tr><th>Time</th><th>Gesture</th><th>Action</th></tr></thead><tbody id="hist"></tbody></table></div>
</div>
<script>
async function tick(){
 try{
  const r=await fetch('/api/status'); const s=await r.json();
  mode.textContent=s.mode;
  lock.textContent=s.locked?'LOCKED':'UNLOCKED'; lock.className='pill '+(s.locked?'off':'on');
  control.textContent=s.control_on?'ON':'OFF'; control.className='pill '+(s.control_on?'on':'off');
  fps.textContent=(s.fps||0).toFixed(1); hands.textContent=s.hands;
  left.textContent=s.left.gesture; leftc.textContent='Confidence '+Math.round(s.left.confidence*100)+'%';
  right.textContent=s.right.gesture; rightc.textContent='Confidence '+Math.round(s.right.confidence*100)+'%';
  action.textContent=s.last_action||'-';
  hist.innerHTML=(s.gesture_history||[]).map(h=>`<tr><td>${h.time}</td><td>${h.gesture}</td><td>${h.action}</td></tr>`).join('');
 }catch(e){}
}
setInterval(tick,500); tick();
</script></body></html>"""


def create_app(state: DashboardState):
    """Build the FastAPI app bound to a DashboardState instance."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="AI Touchless Control Dashboard")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _PAGE

    @app.get("/api/status", response_class=JSONResponse)
    def status():
        return state.snapshot()

    return app


def run_dashboard(state: DashboardState, host="127.0.0.1", port=8000):
    """Start uvicorn in a daemon thread. Returns the thread (or None on failure)."""
    try:
        import uvicorn
    except Exception:
        log.warning("uvicorn not installed; dashboard disabled")
        return None
    app = create_app(state)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    log.info("Dashboard available at http://%s:%d", host, port)
    return t
