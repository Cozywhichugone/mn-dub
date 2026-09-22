#!/usr/bin/env python3
"""Командын мөрөөр дуу оруулах. Олон файл дараалуулахад тохиромжтой.

    python cli.py video.ts -o ./out --voice male --volume 0.08
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pipeline import Options, PipelineError, run

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Англи видеог монгол хоолойтой болгоно.")
    ap.add_argument("video", type=Path, help="Эх видео файл")
    ap.add_argument("-o", "--out", type=Path, default=Path("./out"), help="Гаралтын хавтас")
    ap.add_argument("--voice", choices=["female", "male"], default="female")
    ap.add_argument("--volume", type=float, default=0.10,
                    help="Эх дууны түвшин, 0.0–1.0 (үндсэн 0.10)")
    ap.add_argument("--no-duck", action="store_true",
                    help="Ярианы үед эх дууг автоматаар намсгахгүй")
    ap.add_argument("--burn-subs", action="store_true", help="Хадмалыг видеон дээр шатаах")
    ap.add_argument("--local-whisper", action="store_true",
                    help="Яриа таниулахад локал faster-whisper ашиглах")
    args = ap.parse_args()

    if not args.video.exists():
        print(f"Файл олдсонгүй: {args.video}", file=sys.stderr)
        return 1

    opts = Options(
        voice=args.voice,
        original_volume=args.volume,
        duck=not args.no_duck,
        burn_subtitles=args.burn_subs,
        whisper_backend="local" if args.local_whisper else "api",
        translate_model=os.getenv("TRANSLATE_MODEL", "claude-sonnet-5"),
    )

    keys = {
        "openai": os.getenv("OPENAI_API_KEY", ""),
        "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
        "azure_key": os.getenv("AZURE_SPEECH_KEY", ""),
        "azure_region": os.getenv("AZURE_SPEECH_REGION", ""),
    }

    last = ""

    def progress(step: str, pct: float, message: str) -> None:
        nonlocal last
        if message != last:
            print(f"  [{step:<10}] {message}")
            last = message

    try:
        result = run(args.video, args.out, opts, keys, progress)
    except PipelineError as exc:
        print(f"\nАлдаа: {exc}", file=sys.stderr)
        return 1

    print("\nБэлэн боллоо")
    print(f"  видео : {result['video']}")
    print(f"  хадмал: {result['srt_mn']}")
    print(f"  {result['segments']} өгүүлбэр"
          + (f", {result['compressed']}-ыг хугацаанд нь багтаахаар хурдасгав"
             if result["compressed"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
