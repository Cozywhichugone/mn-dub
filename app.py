"""Монгол дуу оруулгын web интерфэйс."""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file

import pipeline
from pipeline import Options, PipelineError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE = Path(__file__).parent
UPLOADS = BASE / "data" / "uploads"
OUTPUTS = BASE / "data" / "outputs"
for d in (UPLOADS, OUTPUTS):
    d.mkdir(parents=True, exist_ok=True)

ALLOWED = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts", ".m2ts", ".mts",
    ".flv", ".wmv", ".m4v", ".mpg", ".mpeg", ".mp3", ".wav", ".m4a",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_GB", "8")) * 1024**3

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()

STEPS = ["extract", "transcribe", "translate", "synthesize", "mix"]
STEP_WEIGHT = {"extract": 0.04, "transcribe": 0.28, "translate": 0.18,
               "synthesize": 0.36, "mix": 0.14}


def keys_from_env() -> dict[str, str]:
    return {
        "openai": os.getenv("OPENAI_API_KEY", ""),
        "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
        "azure_key": os.getenv("AZURE_SPEECH_KEY", ""),
        "azure_region": os.getenv("AZURE_SPEECH_REGION", ""),
    }


def missing_keys(opts: Options) -> list[str]:
    keys = keys_from_env()
    missing = []
    if opts.whisper_backend == "api" and not keys["openai"]:
        missing.append("OPENAI_API_KEY")
    if not keys["anthropic"]:
        missing.append("ANTHROPIC_API_KEY")
    if not keys["azure_key"]:
        missing.append("AZURE_SPEECH_KEY")
    if not keys["azure_region"]:
        missing.append("AZURE_SPEECH_REGION")
    return missing


def set_job(job_id: str, **fields) -> None:
    with LOCK:
        JOBS[job_id].update(fields)


def worker(job_id: str, video: Path, opts: Options) -> None:
    def progress(step: str, pct: float, message: str) -> None:
        share = max(0.0, min(1.0, pct))
        done = sum(STEP_WEIGHT[s] for s in STEPS[: STEPS.index(step)])
        overall = done + STEP_WEIGHT[step] * share
        with LOCK:
            job = JOBS[job_id]
            job["step"] = step
            job["progress"] = round(overall * 100, 1)
            job["step_progress"] = round(share * 100, 1)
            job["message"] = message
            if not job["log"] or job["log"][-1] != message:
                job["log"].append(message)
                job["log"] = job["log"][-40:]

    set_job(job_id, status="running", started=time.time())
    try:
        result = pipeline.run(video, OUTPUTS / job_id, opts, keys_from_env(), progress)
        set_job(
            job_id,
            status="done",
            progress=100.0,
            message="Бэлэн боллоо",
            result=result,
            finished=time.time(),
        )
    except PipelineError as exc:
        set_job(job_id, status="error", error=str(exc), finished=time.time())
    except Exception as exc:  # noqa: BLE001
        set_job(
            job_id,
            status="error",
            error=f"Санаандгүй алдаа: {exc}",
            finished=time.time(),
        )
    finally:
        try:
            video.unlink(missing_ok=True)
        except OSError:
            pass


@app.get("/")
def index():
    keys = keys_from_env()
    return render_template(
        "index.html",
        ready={
            "openai": bool(keys["openai"]),
            "anthropic": bool(keys["anthropic"]),
            "azure": bool(keys["azure_key"] and keys["azure_region"]),
            "ffmpeg": _ffmpeg_ok(),
        },
    )


def _ffmpeg_ok() -> bool:
    try:
        pipeline.ensure_ffmpeg()
        return True
    except PipelineError:
        return False


@app.post("/api/jobs")
def create_job():
    upload = request.files.get("video")
    if not upload or not upload.filename:
        return jsonify(error="Видео файл сонгоогүй байна."), 400

    ext = Path(upload.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify(
            error=f"{ext or 'Энэ'} өргөтгөлийг дэмжихгүй. "
                  "mp4, mkv, mov, avi, ts, webm зэргийг ашиглана уу."
        ), 400

    glossary = {}
    raw_glossary = (request.form.get("glossary") or "").strip()
    for line in raw_glossary.splitlines():
        if "=" in line:
            src, dst = line.split("=", 1)
            if src.strip() and dst.strip():
                glossary[src.strip()] = dst.strip()

    opts = Options(
        voice=request.form.get("voice", "female"),
        original_volume=float(request.form.get("original_volume", 0.10)),
        duck=request.form.get("duck") == "true",
        burn_subtitles=request.form.get("burn_subtitles") == "true",
        whisper_backend=request.form.get("whisper_backend", "api"),
        translate_model=os.getenv("TRANSLATE_MODEL", "claude-sonnet-5"),
        glossary=glossary,
    )

    absent = missing_keys(opts)
    if absent:
        return jsonify(
            error="Дараах түлхүүрүүд дутуу байна: " + ", ".join(absent)
        ), 400

    job_id = uuid.uuid4().hex[:12]
    safe = re.sub(r"[^\w\-.]", "_", Path(upload.filename).name)[:80]
    dest = UPLOADS / f"{job_id}_{safe}"
    upload.save(dest)

    with LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "filename": upload.filename,
            "status": "queued",
            "step": "extract",
            "progress": 0.0,
            "step_progress": 0.0,
            "message": "Дараалалд орлоо",
            "log": [],
            "result": None,
            "error": None,
            "created": time.time(),
        }

    threading.Thread(target=worker, args=(job_id, dest, opts), daemon=True).start()
    return jsonify(id=job_id), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify(error="Ийм ажил олдсонгүй."), 404
        payload = {k: v for k, v in job.items() if k != "result"}
        if job["result"]:
            payload["result"] = {
                k: v for k, v in job["result"].items() if k != "transcript"
            }
    return jsonify(payload)


@app.get("/api/jobs/<job_id>/transcript")
def transcript(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
    if not job or not job.get("result"):
        return jsonify(error="Бэлэн болоогүй байна."), 404
    return jsonify(job["result"]["transcript"])


@app.get("/api/jobs/<job_id>/download/<kind>")
def download(job_id: str, kind: str):
    with LOCK:
        job = JOBS.get(job_id)
    if not job or not job.get("result"):
        return jsonify(error="Бэлэн болоогүй байна."), 404

    key = {"video": "video", "mn": "srt_mn", "en": "srt_en"}.get(kind)
    if not key:
        return jsonify(error="Ийм файл байхгүй."), 404

    path = Path(job["result"][key])
    if not path.exists():
        return jsonify(error="Файл олдсонгүй."), 404
    return send_file(path, as_attachment=True, download_name=path.name)


@app.get("/api/jobs/<job_id>/preview")
def preview(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
    if not job or not job.get("result"):
        return jsonify(error="Бэлэн болоогүй байна."), 404
    return send_file(Path(job["result"]["video"]), mimetype="video/mp4")


@app.errorhandler(413)
def too_large(_):
    limit = app.config["MAX_CONTENT_LENGTH"] // 1024**3
    return jsonify(error=f"Файл хэт том байна. Дээд хэмжээ {limit}GB."), 413


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"\n  Монгол дуу оруулга → http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, threaded=True)
