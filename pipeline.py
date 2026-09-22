"""
Англи видеог монгол хоолойтой болгох автомат дамжлага.

Дараалал:
  1. ffmpeg      — видеонаас аудио салгах (.ts, .mkv, .mov, .avi бүгд ажиллана)
  2. Whisper     — англи яриаг цаг хугацааны тэмдэглэгээтэй текст болгох
  3. Claude      — сегмент бүрийг контексттэй нь монгол руу орчуулах
  4. Azure TTS   — mn-MN neural хоолойгоор дуу оруулах, хугацаанд нь багтаах
  5. ffmpeg      — эх дууг намсгаж, монгол хоолойг давхарлан mp4 болгох

Гадаад хамаарал: ffmpeg, requests. Бусад нь стандарт сан.
"""

from __future__ import annotations

import array
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import requests

# ---------------------------------------------------------------- тохиргоо

SAMPLE_RATE = 24_000          # Azure-ийн riff-24khz-16bit-mono-pcm-тэй тааруулав
WHISPER_CHUNK_SECONDS = 600   # 25MB хязгаарт багтаахын тулд 10 минутаар хуваана
MAX_SPEEDUP = 1.35            # Үүнээс хурдан яривал сонсоход эвгүй болно
TRANSLATE_BATCH = 40          # Нэг API дуудалтад орох сегментийн тоо

MN_VOICES = {
    "female": "mn-MN-YesuiNeural",
    "male": "mn-MN-BataaNeural",
}


class PipelineError(RuntimeError):
    """Хэрэглэгчид харуулахад тохиромжтой алдаа."""


@dataclass
class Segment:
    index: int
    start: float
    end: float
    source: str
    target: str = ""
    audio: Path | None = None
    rendered: float = 0.0          # синтезчилсэн аудионы бодит урт
    speed: float = 1.0             # хэрэглэсэн prosody rate

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Options:
    voice: str = "female"
    original_volume: float = 0.10  # эх дууны үлдээх хэмжээ (0 = бүрэн дарах)
    duck: bool = True              # яриа орох үед эх дууг автоматаар намсгах
    burn_subtitles: bool = False
    source_language: str = "en"
    translate_model: str = "claude-sonnet-5"
    whisper_backend: str = "api"   # "api" эсвэл "local"
    keep_workdir: bool = False
    glossary: dict[str, str] = field(default_factory=dict)


Progress = Callable[[str, float, str], None]  # (алхам, 0..1, мессеж)


def _noop(step: str, pct: float, message: str) -> None:
    pass


# ---------------------------------------------------------------- туслахууд

def _run(cmd: list[str], what: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-6:]
        raise PipelineError(f"{what} амжилтгүй боллоо:\n" + "\n".join(tail))
    return proc


def ensure_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise PipelineError(
                f"{tool} олдсонгүй. ffmpeg суулгана уу: "
                "https://ffmpeg.org/download.html"
            )


def media_duration(path: Path) -> float:
    proc = _run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        "Файлын урт тодорхойлох",
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        raise PipelineError("Файлын үргэлжлэх хугацааг уншиж чадсангүй.")


def has_audio_stream(path: Path) -> bool:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True,
    )
    return bool(proc.stdout.strip())


# ---------------------------------------------------- 1. аудио салгах

def extract_audio(video: Path, out_wav: Path, progress: Progress = _noop) -> Path:
    progress("extract", 0.0, "Видеонаас аудио салгаж байна")
    if not has_audio_stream(video):
        raise PipelineError("Энэ видеонд аудио суваг алга байна.")
    _run(
        [
            "ffmpeg", "-y", "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(out_wav),
        ],
        "Аудио салгах",
    )
    progress("extract", 1.0, "Аудио бэлэн")
    return out_wav


# ---------------------------------------------------- 2. Whisper

def _split_for_whisper(wav: Path, workdir: Path) -> list[tuple[Path, float]]:
    """25MB хязгаараас хэтэрвэл хэсэглэнэ. (файл, эхлэх offset) буцаана."""
    if wav.stat().st_size < 24 * 1024 * 1024:
        return [(wav, 0.0)]

    chunk_dir = workdir / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    _run(
        [
            "ffmpeg", "-y", "-i", str(wav),
            "-f", "segment", "-segment_time", str(WHISPER_CHUNK_SECONDS),
            "-c", "copy", str(chunk_dir / "part%04d.wav"),
        ],
        "Аудио хэсэглэх",
    )
    parts = sorted(chunk_dir.glob("part*.wav"))
    out: list[tuple[Path, float]] = []
    offset = 0.0
    for part in parts:
        out.append((part, offset))
        offset += media_duration(part)
    return out


def transcribe(
    wav: Path,
    workdir: Path,
    opts: Options,
    openai_key: str | None,
    progress: Progress = _noop,
) -> list[Segment]:
    progress("transcribe", 0.0, "Англи яриаг таньж байна")

    if opts.whisper_backend == "local":
        segments = _transcribe_local(wav, opts, progress)
    else:
        if not openai_key:
            raise PipelineError("OPENAI_API_KEY тохируулаагүй байна.")
        segments = _transcribe_api(wav, workdir, opts, openai_key, progress)

    if not segments:
        raise PipelineError("Яриа илрээгүй. Аудио чимээгүй эсвэл хэт бүдэг байж магадгүй.")

    progress("transcribe", 1.0, f"{len(segments)} өгүүлбэр таньсан")
    return segments


def _transcribe_api(
    wav: Path, workdir: Path, opts: Options, key: str, progress: Progress
) -> list[Segment]:
    chunks = _split_for_whisper(wav, workdir)
    segments: list[Segment] = []

    for i, (part, offset) in enumerate(chunks):
        progress(
            "transcribe",
            i / max(1, len(chunks)),
            f"Whisper: {i + 1}/{len(chunks)} хэсэг",
        )
        with part.open("rb") as fh:
            resp = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (part.name, fh, "audio/wav")},
                data={
                    "model": "whisper-1",
                    "language": opts.source_language,
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "segment",
                },
                timeout=600,
            )
        if resp.status_code != 200:
            raise PipelineError(f"Whisper алдаа {resp.status_code}: {resp.text[:300]}")

        for seg in resp.json().get("segments", []):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            segments.append(
                Segment(
                    index=len(segments),
                    start=float(seg["start"]) + offset,
                    end=float(seg["end"]) + offset,
                    source=text,
                )
            )
    return segments


def _transcribe_local(wav: Path, opts: Options, progress: Progress) -> list[Segment]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise PipelineError(
            "Локал горимд faster-whisper хэрэгтэй: pip install faster-whisper"
        )

    model_size = os.getenv("WHISPER_MODEL", "large-v3")
    device = os.getenv("WHISPER_DEVICE", "auto")
    compute = os.getenv("WHISPER_COMPUTE", "int8")
    progress("transcribe", 0.05, f"Whisper {model_size} ачаалж байна")

    model = WhisperModel(model_size, device=device, compute_type=compute)
    result, info = model.transcribe(str(wav), language=opts.source_language, vad_filter=True)

    segments: list[Segment] = []
    total = info.duration or 1.0
    for seg in result:
        text = (seg.text or "").strip()
        if text:
            segments.append(
                Segment(index=len(segments), start=seg.start, end=seg.end, source=text)
            )
        progress("transcribe", min(0.99, seg.end / total), f"{len(segments)} өгүүлбэр")
    return segments


# ---------------------------------------------------- 3. орчуулга

TRANSLATE_SYSTEM = """Чи англи хэлнээс монгол руу орчуулдаг мэргэжлийн орчуулагч. \
Энэ бол видеоны дуу оруулгын скрипт тул орчуулга чинь чанга уншихад \
байгалийн, ярианы аястай сонсогдох ёстой.

Дүрэм:
- Утгыг нь хадгал, үг үсгээр нь бүү орчуул. Англи хэлний өгүүлбэрийн бүтцийг бүү даган хуул.
- Монгол хүн ярьж байгаа юм шиг эгшиглэ. Албархуу, хэвшмэл орчуулгын хэлээс зайлсхий.
- Уншихад орох хугацаа нь эхтэйгээ ойролцоо байхаар урт нь тэнцүү орчим байг. \
Аль болох эх өгүүлбэрээс уртасгахгүй бай.
- Тоо, огноо, хэмжигдэхүүнийг цифрээр бус, уншиж хэлдэг хэлбэрээр бич \
(жишээ нь "25%" биш "хорин таван хувь").
- Хүн, байгууллага, бүтээгдэхүүний нэрийг монгол галигаар бич.
- Сегментүүд нэг үргэлжилсэн яриа тул өмнөх, дараах өгүүлбэртэй утгаараа зохицуул.
- Тайлбар, тэмдэглэл бүү нэм.

Гаралт: зөвхөн JSON массив, өөр юу ч бичихгүй. Элемент бүр {"id": <тоо>, "mn": "<орчуулга>"}."""


def translate(
    segments: list[Segment],
    opts: Options,
    anthropic_key: str,
    progress: Progress = _noop,
) -> list[Segment]:
    if not anthropic_key:
        raise PipelineError("ANTHROPIC_API_KEY тохируулаагүй байна.")

    progress("translate", 0.0, "Монгол руу орчуулж байна")
    batches = [
        segments[i: i + TRANSLATE_BATCH]
        for i in range(0, len(segments), TRANSLATE_BATCH)
    ]

    glossary_note = ""
    if opts.glossary:
        pairs = "\n".join(f"- {k} → {v}" for k, v in opts.glossary.items())
        glossary_note = f"\n\nЭдгээр нэр томьёог заавал ингэж орчуул:\n{pairs}"

    for bi, batch in enumerate(batches):
        before = segments[max(0, batch[0].index - 2): batch[0].index]
        after = segments[batch[-1].index + 1: batch[-1].index + 3]

        context = ""
        if before:
            context += "Өмнөх хэсэг: " + " ".join(s.source for s in before) + "\n"
        if after:
            context += "Дараах хэсэг: " + " ".join(s.source for s in after) + "\n"

        payload = [{"id": s.index, "en": s.source} for s in batch]
        user_msg = (
            f"{context}\nОрчуулах сегментүүд:\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=1)}"
        )

        text = _anthropic_call(
            anthropic_key,
            opts.translate_model,
            TRANSLATE_SYSTEM + glossary_note,
            user_msg,
        )
        mapping = _parse_translation(text, batch)

        for seg in batch:
            seg.target = mapping.get(seg.index, "").strip()

        progress(
            "translate",
            (bi + 1) / len(batches),
            f"Орчуулга: {min((bi + 1) * TRANSLATE_BATCH, len(segments))}/{len(segments)}",
        )

    missing = [s for s in segments if not s.target]
    if len(missing) > len(segments) * 0.2:
        raise PipelineError("Орчуулгын ихэнх хэсэг буцаж ирсэнгүй. Дахин оролдоно уу.")

    progress("translate", 1.0, "Орчуулга бэлэн")
    return segments


def _anthropic_call(key: str, model: str, system: str, user: str, retries: int = 3) -> str:
    last = ""
    for attempt in range(retries):
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 8000,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=300,
        )
        if resp.status_code == 200:
            return "".join(
                blk.get("text", "")
                for blk in resp.json().get("content", [])
                if blk.get("type") == "text"
            )
        last = f"{resp.status_code}: {resp.text[:300]}"
        if resp.status_code not in (429, 500, 502, 503, 529):
            break
    raise PipelineError(f"Орчуулгын API алдаа {last}")


def _parse_translation(text: str, batch: list[Segment]) -> dict[int, str]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    out: dict[int, str] = {}
    valid = {s.index for s in batch}
    items = data if isinstance(data, list) else []

    for position, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        idx = None
        for key in ("id", "index", "i"):
            if key in item:
                try:
                    idx = int(item[key])
                except (TypeError, ValueError):
                    idx = None
                break
        # Дугаараа алдсан ч дараалал нь зөв бол байрлалаар нь сэргээнэ
        if idx is None and position < len(batch):
            idx = batch[position].index

        text = ""
        for key in ("mn", "target", "translation", "text"):
            if key in item:
                text = str(item[key]).strip()
                break

        if idx in valid and text:
            out[idx] = text
    return out


# ---------------------------------------------------- 4. Azure TTS

def _ssml(text: str, voice: str, rate: float) -> str:
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    rate_attr = f'{round((rate - 1) * 100)}%'
    if not rate_attr.startswith("-"):
        rate_attr = "+" + rate_attr
    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        'xml:lang="mn-MN">'
        f'<voice name="{voice}">'
        f'<prosody rate="{rate_attr}">{safe}</prosody>'
        "</voice></speak>"
    )


def _azure_tts(text: str, voice: str, rate: float, key: str, region: str, out: Path) -> float:
    resp = requests.post(
        f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
        headers={
            "Ocp-Apim-Subscription-Key": key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
            "User-Agent": "mn-dub",
        },
        data=_ssml(text, voice, rate).encode("utf-8"),
        timeout=120,
    )
    if resp.status_code != 200:
        raise PipelineError(
            f"Azure TTS алдаа {resp.status_code}: {resp.text[:200] or 'хариу хоосон'}"
        )
    out.write_bytes(resp.content)
    return _wav_duration(out)


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def synthesize(
    segments: list[Segment],
    workdir: Path,
    opts: Options,
    azure_key: str,
    azure_region: str,
    progress: Progress = _noop,
) -> list[Segment]:
    if not azure_key or not azure_region:
        raise PipelineError("AZURE_SPEECH_KEY болон AZURE_SPEECH_REGION хэрэгтэй.")

    voice = MN_VOICES.get(opts.voice, MN_VOICES["female"])
    voice_dir = workdir / "voice"
    voice_dir.mkdir(exist_ok=True)
    progress("synthesize", 0.0, "Монгол хоолой үүсгэж байна")

    for i, seg in enumerate(segments):
        if not seg.target:
            continue

        # Дараагийн сегмент хүртэлх бодит зай — үүнд багтаах ёстой
        if i + 1 < len(segments):
            budget = max(seg.duration, segments[i + 1].start - seg.start - 0.05)
        else:
            budget = seg.duration + 1.5

        out = voice_dir / f"seg{seg.index:05d}.wav"
        length = _azure_tts(seg.target, voice, 1.0, azure_key, azure_region, out)

        # Хугацаанд багтахгүй бол хурдыг нэмж дахин уншуулна
        if budget > 0.2 and length > budget * 1.02:
            needed = min(MAX_SPEEDUP, length / budget)
            length = _azure_tts(seg.target, voice, needed, azure_key, azure_region, out)
            seg.speed = needed

        seg.audio = out
        seg.rendered = length
        progress(
            "synthesize",
            (i + 1) / len(segments),
            f"Хоолой: {i + 1}/{len(segments)}",
        )

    progress("synthesize", 1.0, "Хоолой бэлэн")
    return segments


# ---------------------------------------------------- 5. угсрах, холих

def assemble_track(segments: list[Segment], total: float, out_wav: Path) -> Path:
    """Сегментүүдийг цагийн хуваарьт нь тавьж нэг аудио суваг болгоно."""
    frames = int(math.ceil((total + 2.0) * SAMPLE_RATE))
    track = array.array("h", bytes(frames * 2))

    gap = int(0.06 * SAMPLE_RATE)   # сегмент хооронд багахан завсар
    cursor = 0                      # өмнөх сегмент дуусаж буй байрлал

    for seg in sorted(segments, key=lambda s: s.start):
        if not seg.audio or not seg.audio.exists():
            continue
        with wave.open(str(seg.audio), "rb") as wf:
            data = array.array("h", wf.readframes(wf.getnframes()))

        # Өмнөх сегмент слотоосоо халисан бол хоёр хоолой зэрэг
        # сонсогдохоос сэргийлж эхлэлийг нь хойшлуулна.
        offset = max(int(seg.start * SAMPLE_RATE), cursor)
        cursor = offset + len(data) + gap

        for j, sample in enumerate(data):
            pos = offset + j
            if pos >= frames:
                break
            # Давхцвал нэмж, хязгаараас хэтрэхээс сэргийлнэ
            mixed = track[pos] + sample
            track[pos] = 32767 if mixed > 32767 else (-32768 if mixed < -32768 else mixed)

    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(track.tobytes())
    return out_wav


def _srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[Segment], out: Path, field_name: str = "target") -> Path:
    lines: list[str] = []
    n = 0
    for seg in segments:
        text = getattr(seg, field_name)
        if not text:
            continue
        n += 1
        lines.append(str(n))
        lines.append(f"{_srt_time(seg.start)} --> {_srt_time(seg.end)}")
        lines.append(text)
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def mux(
    video: Path,
    dub_wav: Path,
    out_video: Path,
    opts: Options,
    subtitle: Path | None = None,
) -> Path:
    if opts.duck:
        # Монгол хоолой орох үед эх дууг автоматаар намсгана.
        # Хоолойн замыг asplit-ээр хоёр хуваана: нэг нь удирдах дохио,
        # нөгөө нь эцсийн холимогт орно.
        chain = (
            f"[0:a]volume={max(opts.original_volume, 0.25)},"
            "aformat=channel_layouts=stereo[bg];"
            "[1:a]aformat=channel_layouts=stereo,volume=1.6,asplit=2[vo][key];"
            "[bg][key]sidechaincompress="
            "threshold=0.02:ratio=14:attack=8:release=380[ducked];"
            "[ducked][vo]amix=inputs=2:duration=first:normalize=0[aout]"
        )
    else:
        chain = (
            f"[0:a]volume={opts.original_volume},aformat=channel_layouts=stereo[bg];"
            "[1:a]aformat=channel_layouts=stereo,volume=1.6[vo];"
            "[bg][vo]amix=inputs=2:duration=first:normalize=0[aout]"
        )

    cmd = ["ffmpeg", "-y", "-i", str(video), "-i", str(dub_wav)]

    if subtitle and opts.burn_subtitles:
        escaped = str(subtitle).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
        style = (
            "FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H90000000,BorderStyle=3,Outline=1,MarginV=28"
        )
        chain += f";[0:v]subtitles='{escaped}':force_style='{style}'[vout]"
        video_map = ["-map", "[vout]", "-c:v", "libx264", "-crf", "20", "-preset", "medium"]
    else:
        video_map = ["-map", "0:v", "-c:v", "copy"]

    cmd += [
        "-filter_complex", chain,
        *video_map,
        "-map", "[aout]",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(out_video),
    ]
    _run(cmd, "Видео угсрах")
    return out_video


# ---------------------------------------------------- бүх дамжлага

def run(
    video: Path,
    outdir: Path,
    opts: Options,
    keys: dict[str, str],
    progress: Progress = _noop,
) -> dict:
    ensure_ffmpeg()
    video = Path(video)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    workdir = Path(tempfile.mkdtemp(prefix="mndub_"))
    stem = re.sub(r"[^\w\-.]", "_", video.stem)[:60] or "video"

    try:
        total = media_duration(video)

        wav = extract_audio(video, workdir / "source.wav", progress)
        segments = transcribe(wav, workdir, opts, keys.get("openai"), progress)
        segments = translate(segments, opts, keys.get("anthropic", ""), progress)
        segments = synthesize(
            segments, workdir, opts,
            keys.get("azure_key", ""), keys.get("azure_region", ""),
            progress,
        )

        progress("mix", 0.15, "Дууны замыг угсарч байна")
        dub = assemble_track(segments, total, workdir / "dub.wav")

        srt_mn = write_srt(segments, outdir / f"{stem}.mn.srt", "target")
        srt_en = write_srt(segments, outdir / f"{stem}.en.srt", "source")

        progress("mix", 0.45, "Видеотой нийлүүлж байна")
        out_video = mux(video, dub, outdir / f"{stem}.mn.mp4", opts, srt_mn)
        progress("mix", 1.0, "Дууссан")

        sped = sum(1 for s in segments if s.speed > 1.01)
        return {
            "video": str(out_video),
            "srt_mn": str(srt_mn),
            "srt_en": str(srt_en),
            "segments": len(segments),
            "duration": total,
            "compressed": sped,
            "transcript": [
                {"start": s.start, "end": s.end, "en": s.source, "mn": s.target}
                for s in segments
            ],
        }
    finally:
        if not opts.keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
