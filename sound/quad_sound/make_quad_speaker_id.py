#!/usr/bin/env python3
"""Build a quadraphonic spatial-granularity test file.

Voices are placed evenly around the four-speaker square and play in
order, one after another, like the original speaker-ID file. Corners
are a single speaker; positions between corners are a pan across the
two adjacent speakers.

Clockwise around the room (not WAV channel index order — that would
put some "in-between" voices on a diagonal):

    Front Left → Front Right → Back Right → Back Left → (wrap)

    FL ---- FR
    |        |
    BL ---- BR

Channel order in the file is still FFmpeg `quad`: FL FR BL BR.

`--count` / `-n` is how many distinct voices to place. Examples:

    4  — one voice per speaker (same idea as the original ID file,
         but clockwise: FL, FR, BR, BL)
    8  — a midpoint between each pair as well
         (noise 1 = FL, noise 2 = 50/50 FL+FR, noise 3 = FR, …)
    12 — two steps between each pair
    16 — three steps between each pair

Default pan is equal-power (midpoints stay as loud as corners).
`--pan linear` uses amplitude 1−t / t instead (a midpoint is
literally 50% / 50% on the two channels, and a bit quieter).

Uses macOS `say` for TTS and ffmpeg to mux a 4-channel file.
Default output is FLAC. WAV (24-bit PCM) is also supported.

Examples (from the repo root):

    python3 sound/quad_sound/make_quad_speaker_id.py

        8 voices, equal-power, sound/quad_sound/quad_granularity_8.flac

    python3 sound/quad_sound/make_quad_speaker_id.py -n 4

        Corners only.

    python3 sound/quad_sound/make_quad_speaker_id.py -n 12 --gap 1.0

        12 voices, 1s pause between them. Listen for where adjacent
        numbers start to come from the same place.

    python3 sound/quad_sound/make_quad_speaker_id.py -n 8 --pan linear

        Midpoints are an even 50/50 amplitude split.

    python3 sound/quad_sound/make_quad_speaker_id.py -n 16 --format wav \\
        -o sound/quad_sound/granularity_16.wav --voice Samantha

        Named macOS voice. List voices with:  say -v '?'
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# File / FFmpeg quad channel order
SPEAKERS = (
    ("Front Left", "FL"),
    ("Front Right", "FR"),
    ("Back Left", "BL"),
    ("Back Right", "BR"),
)

# Clockwise walk around the square, as indices into SPEAKERS
CYCLE = (0, 1, 3, 2)  # FL, FR, BR, BL

SAMPLE_RATE = 48000
DEFAULT_GAP_S = 0.75
DEFAULT_COUNT = 8
SCRIPT_DIR = Path(__file__).resolve().parent

_ONES = (
    "zero one two three four five six seven eight nine ten "
    "eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen"
).split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()


@dataclass(frozen=True)
class Voice:
    index: int  # 1-based
    text: str
    gains: tuple[float, float, float, float]  # FL, FR, BL, BR
    where: str


def spoken_number(n: int) -> str:
    if n < 20:
        return _ONES[n].capitalize()
    if n < 100:
        ten, one = divmod(n, 10)
        word = _TENS[ten - 2]
        if one:
            word += f"-{_ONES[one]}"
        return word.capitalize()
    return str(n)


def voice_at(i: int, count: int, pan: str) -> Voice:
    """Place voice i (0-based) of `count` evenly around the square."""
    n_edges = len(CYCLE)
    t = (i * n_edges) / count
    edge = int(math.floor(t)) % n_edges
    frac = t - math.floor(t)
    a = CYCLE[edge]
    b = CYCLE[(edge + 1) % n_edges]

    gains = [0.0, 0.0, 0.0, 0.0]
    if frac < 1e-9:
        gains[a] = 1.0
        where = SPEAKERS[a][0]
    elif frac > 1.0 - 1e-9:
        gains[b] = 1.0
        where = SPEAKERS[b][0]
    else:
        if pan == "equal-power":
            gains[a] = math.cos(frac * math.pi / 2)
            gains[b] = math.sin(frac * math.pi / 2)
        else:
            gains[a] = 1.0 - frac
            gains[b] = frac
        where = f"{SPEAKERS[a][0]} and {SPEAKERS[b][0]}"

    index = i + 1
    text = f"{spoken_number(index)}. {where}"
    return Voice(index, text, tuple(gains), where)


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        sys.exit(f"Required tool not found on PATH: {tool}")
    return path


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(" ", " ".join(cmd))
    return subprocess.run(cmd, check=True, **kwargs)


def synthesize(say_bin: str, text: str, out_wav: Path, voice: str | None) -> None:
    cmd = [
        say_bin,
        "--file-format=WAVE",
        f"--data-format=LEI16@{SAMPLE_RATE}",
        "-o",
        str(out_wav),
    ]
    if voice:
        cmd.extend(["-v", voice])
    cmd.append(text)
    _run(cmd)


def probe_duration(ffprobe_bin: str, path: Path) -> float:
    result = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _pan_filter(gains: tuple[float, float, float, float]) -> str:
    assigns = "|".join(
        f"{abbr}={gain:.6f}*c0" for (_name, abbr), gain in zip(SPEAKERS, gains)
    )
    return f"pan=quad|{assigns}"


def mux_quad(
    ffmpeg_bin: str,
    clips: list[Path],
    voices: list[Voice],
    durations: list[float],
    gap_s: float,
    output: Path,
    audio_codec: str,
) -> None:
    starts = []
    cursor = 0.0
    for i, dur in enumerate(durations):
        starts.append(cursor)
        cursor += dur
        if i < len(durations) - 1:
            cursor += gap_s
    total_s = cursor + gap_s

    filters = []
    labels = []
    for i, (start, voice) in enumerate(zip(starts, voices)):
        delay_ms = int(round(start * 1000))
        label = f"p{i}"
        filters.append(
            f"[{i}:a]adelay=delays={delay_ms}:all=1,"
            f"apad=whole_dur={total_s:.6f},"
            f"{_pan_filter(voice.gains)}[{label}]"
        )
        labels.append(f"[{label}]")

    n = len(clips)
    if n == 1:
        filters.append(f"{labels[0]}acopy[a]")
    else:
        filters.append(
            "".join(labels)
            + f"amix=inputs={n}:duration=longest:dropout_transition=0:normalize=0[a]"
        )

    cmd = [ffmpeg_bin, "-y"]
    for clip in clips:
        cmd.extend(["-i", str(clip)])
    cmd.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[a]",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            audio_codec,
            "-channel_layout",
            "quad",
            str(output),
        ]
    )
    _run(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a quadraphonic TTS file with N voices spaced evenly "
            "around the four speakers."
        )
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=(
            "Number of voices around the square "
            f"(default: {DEFAULT_COUNT}). 4 = corners only; "
            "8 = a midpoint between each pair."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path (.flac or .wav). "
            f"Default: sound/quad_sound/quad_granularity_<count>.flac"
        ),
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=DEFAULT_GAP_S,
        help=f"Silence between voices in seconds (default: {DEFAULT_GAP_S})",
    )
    parser.add_argument(
        "--pan",
        choices=("equal-power", "linear"),
        default="equal-power",
        help=(
            "How to split a voice across two speakers "
            "(default: equal-power). linear = amplitude 50/50 at midpoints."
        ),
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="macOS `say` voice name (default: system voice)",
    )
    parser.add_argument(
        "--format",
        choices=("wav", "flac"),
        default=None,
        help="Container/codec. Inferred from --output suffix if omitted.",
    )
    return parser.parse_args()


def _gains_summary(gains: tuple[float, float, float, float]) -> str:
    parts = [
        f"{abbr}={gain:.2f}"
        for (_name, abbr), gain in zip(SPEAKERS, gains)
        if gain > 1e-6
    ]
    return " ".join(parts)


def main() -> None:
    if sys.platform != "darwin":
        sys.exit("This script uses macOS `say` for TTS and currently runs only on macOS.")

    args = parse_args()
    if args.count < 1:
        sys.exit("--count must be at least 1")

    voices = [voice_at(i, args.count, args.pan) for i in range(args.count)]

    if args.output is None:
        output = SCRIPT_DIR / f"quad_granularity_{args.count}.flac"
    else:
        output = args.output.expanduser().resolve()
    suffix = output.suffix.lower().lstrip(".")
    fmt = args.format or (suffix if suffix in {"wav", "flac"} else "flac")
    if output.suffix.lower() != f".{fmt}":
        output = output.with_suffix(f".{fmt}")

    audio_codec = "flac" if fmt == "flac" else "pcm_s24le"

    say_bin = _require("say")
    ffmpeg_bin = _require("ffmpeg")
    ffprobe_bin = _require("ffprobe")

    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Placing {args.count} voices around the square ({args.pan} pan):")
    for voice in voices:
        print(f"  {voice.index:3d}  {voice.text:<45s}  {_gains_summary(voice.gains)}")

    print("Synthesizing with macOS say …")
    with tempfile.TemporaryDirectory(prefix="quad_granularity_") as tmp:
        tmp_dir = Path(tmp)
        clips: list[Path] = []
        durations: list[float] = []
        for voice in voices:
            clip = tmp_dir / f"{voice.index:03d}.wav"
            synthesize(say_bin, voice.text, clip, args.voice)
            dur = probe_duration(ffprobe_bin, clip)
            clips.append(clip)
            durations.append(dur)
            print(f"      {dur:.2f}s")

        print(f"Muxing 4-channel {fmt.upper()} → {output}")
        mux_quad(ffmpeg_bin, clips, voices, durations, args.gap, output, audio_codec)

    print("Done.")
    print(f"  Layout: FL FR BL BR  (FFmpeg quad)")
    print(f"  Cycle:  FL → FR → BR → BL")
    print(f"  Voices: {args.count}  pan={args.pan}  gap={args.gap}s")
    print(f"  File:   {output}")


if __name__ == "__main__":
    main()
