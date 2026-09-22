import os, shutil, threading, time
from flask import Response, request
from app import app
try:
    from app import JOBS, LOCK, OUTPUTS, UPLOADS
except ImportError:
    JOBS, LOCK, OUTPUTS, UPLOADS = {}, threading.Lock(), None, None

APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()
MAX_AGE_HOURS = float(os.getenv("JOB_MAX_AGE_HOURS", "6"))

@app.before_request
def _require_login():
    if not APP_PASSWORD:
        return None
    auth = request.authorization
    if auth and auth.password == APP_PASSWORD:
        return None
    return Response("Nevtreh shaardlagatai.", 401,
                    {"WWW-Authenticate": "Basic realm=mn-dub"})

def _reap_once():
    cutoff = time.time() - MAX_AGE_HOURS * 3600
    with LOCK:
        for jid in [j for j, v in JOBS.items() if v.get("created", 0) < cutoff]:
            JOBS.pop(jid, None)
    for folder in (UPLOADS, OUTPUTS):
        if not folder or not folder.exists():
            continue
        for item in folder.iterdir():
            try:
                if item.stat().st_mtime >= cutoff:
                    continue
                shutil.rmtree(item) if item.is_dir() else item.unlink()
            except OSError:
                pass

def _reaper():
    while True:
        time.sleep(1800)
        try:
            _reap_once()
        except Exception:
            pass

threading.Thread(target=_reaper, daemon=True).start()
print("APP_PASSWORD: " + ("ON" if APP_PASSWORD else "OFF - huudas neelttei!"), flush=True)
