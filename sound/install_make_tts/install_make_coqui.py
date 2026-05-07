"""
Coqui VITS TTS batch runner.

Reads all CSVs from the `input_csvs/` folder next to this script, generates
WAV files with Coqui TTS (VCTK multi-speaker VITS), and maintains the same
`have_barked.csv` deduplication log so already-processed image_ids are skipped
across runs and across Bark/Coqui jobs.

Score range: 0.6 <= topic_fit < 0.65
Speaker:     random VCTK speaker picked per line (109 available)
Output:      tts_bark_out/  (same dir as Bark — picked up by batch_collector.py unchanged)
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional, Set

import torch


# ── Logging noise suppression ─────────────────────────────────────────────────

class _SuppressCoquiNoise(logging.Filter):
    _PATTERNS = ("coqpit", "config", "model", "loading", "setting")
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage().lower()
        return not any(p in msg for p in self._PATTERNS)

for _logger_name in ("TTS", "TTS.tts", "TTS.utils", "coqpit"):
    logging.getLogger(_logger_name).addFilter(_SuppressCoquiNoise())


# ── Paths & constants ─────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))

IN_CSV_DIR      = os.path.join(_HERE, "input_csvs")
OUT_DIR         = os.path.join(_HERE, "tts_bark_out")   # shared with Bark
HAVE_BARKED_CSV = os.path.join(_HERE, "have_barked.csv")

TOPIC_FIT_FIELD = "topic_fit"
TOPIC_FIT_MIN   = 0.6
TOPIC_FIT_MAX   = 0.65

MAX_PROCESSED = 0  # 0 = no limit

# Full VCTK speaker list for tts_models/en/vctk/vits
# Used as fallback if tts.speakers is unavailable
VCTK_SPEAKERS = [
    "p225","p226","p227","p228","p229","p230","p231","p232","p233","p234",
    "p236","p237","p238","p239","p240","p241","p243","p244","p245","p246",
    "p247","p248","p249","p250","p251","p252","p253","p254","p255","p256",
    "p257","p258","p259","p260","p261","p262","p263","p264","p265","p266",
    "p267","p268","p269","p270","p271","p272","p273","p274","p275","p276",
    "p277","p278","p279","p280","p281","p282","p283","p284","p285","p286",
    "p287","p288","p292","p293","p294","p295","p297","p298","p299","p300",
    "p301","p302","p303","p304","p305","p306","p307","p308","p310","p311",
    "p312","p313","p314","p316","p317","p318","p323","p326","p329","p330",
    "p333","p334","p335","p336","p339","p340","p341","p343","p345","p347",
    "p351","p360","p361","p362","p363","p364","p374","p376",
]


# ── CSV helpers (identical to Bark script) ────────────────────────────────────

def _safe_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        s = str(value).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


def _load_have_barked_ids(have_barked_csv: str) -> Set[int]:
    if not os.path.exists(have_barked_csv):
        return set()
    ids: Set[int] = set()
    with open(have_barked_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "image_id" in reader.fieldnames:
            for row in reader:
                image_id = _safe_int(row.get("image_id"))
                if image_id is not None:
                    ids.add(image_id)
        else:
            f.seek(0)
            raw = csv.reader(f)
            for r in raw:
                if not r:
                    continue
                image_id = _safe_int(r[0])
                if image_id is not None:
                    ids.add(image_id)
    return ids


def _append_have_barked_id(have_barked_csv: str, image_id: int) -> None:
    exists = os.path.exists(have_barked_csv)
    os.makedirs(os.path.dirname(os.path.abspath(have_barked_csv)) or ".", exist_ok=True)
    with open(have_barked_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id"])
        if not exists:
            writer.writeheader()
        writer.writerow({"image_id": image_id})


def _collect_input_csvs(csv_dir: str) -> list[str]:
    if not os.path.isdir(csv_dir):
        raise FileNotFoundError(
            f"input_csvs folder not found: {csv_dir}\n"
            "Create it and place your CSV files inside before running."
        )
    paths = sorted(
        os.path.join(csv_dir, f)
        for f in os.listdir(csv_dir)
        if f.lower().endswith(".csv")
    )
    if not paths:
        raise FileNotFoundError(f"No .csv files found in {csv_dir}")
    return paths


def _iter_rows(input_csv: str) -> Iterable[dict]:
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _prescan_csvs(input_csvs: list[str], image_id_field: str) -> tuple[int, int]:
    """Return (total_rows, total_in_topic_fit) across all input CSVs."""
    total_rows = 0
    total_in_topic_fit = 0
    for path in input_csvs:
        for row in _iter_rows(path):
            if _safe_int(row.get(image_id_field)) is None:
                continue
            total_rows += 1
            fit_raw = row.get(TOPIC_FIT_FIELD)
            try:
                fit = float(fit_raw) if fit_raw is not None and str(fit_raw).strip() != "" else None
            except Exception:
                fit = None
            if fit is not None and TOPIC_FIT_MIN <= fit < TOPIC_FIT_MAX:
                total_in_topic_fit += 1
    return total_rows, total_in_topic_fit


# ── CoquiVITS wrapper ─────────────────────────────────────────────────────────

@dataclass
class CoquiVITS:
    """
    Thin wrapper around Coqui TTS VCTK-VITS.

    VITS is non-autoregressive — inference on short texts is fast (~50-150ms
    per line on a 4090). Each line gets a freshly random speaker from the
    full 109-speaker VCTK set.
    """
    _tts: object       # TTS instance — untyped to avoid import-time dep
    sample_rate: int
    speaker_list: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, device: Optional[str] = None) -> "CoquiVITS":
        from TTS.api import TTS  # pip install TTS

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading Coqui VCTK-VITS on {device} …")
        tts = TTS(
            model_name="tts_models/en/vctk/vits",
            progress_bar=False,
            gpu=(device == "cuda"),
        )

        # Prefer the live speaker list from the loaded model
        try:
            speakers = list(tts.speakers) if tts.speakers else VCTK_SPEAKERS
        except Exception:
            speakers = VCTK_SPEAKERS

        sample_rate = 22050
        try:
            sample_rate = tts.synthesizer.output_sample_rate
        except Exception:
            pass

        print(f"Coqui VCTK-VITS ready. {len(speakers)} speakers. "
              f"Sample rate: {sample_rate} Hz")
        return cls(_tts=tts, sample_rate=sample_rate, speaker_list=speakers)

    def synthesize_to_wav(self, text: str, out_wav_path: str, speaker: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(out_wav_path)) or ".", exist_ok=True)
        self._tts.tts_to_file(text=text, speaker=speaker, file_path=out_wav_path)
        return out_wav_path

    def random_speaker(self) -> str:
        return random.choice(self.speaker_list)


# ── Output path ───────────────────────────────────────────────────────────────

def _build_out_path(out_dir: str, image_id: int, speaker: str) -> str:
    filename = f"{image_id}_coqui_{speaker}.wav"
    return os.path.join(out_dir, filename)


# ── Pending item ──────────────────────────────────────────────────────────────

@dataclass
class _PendingItem:
    image_id: int
    text: str
    out_path: str
    speaker: str


# ── Flush ─────────────────────────────────────────────────────────────────────

def _flush_batch(
    tts: CoquiVITS,
    pending: list[_PendingItem],
    already: Set[int],
) -> tuple[int, list[str]]:
    if not pending:
        return 0, []

    written: list[str] = []
    for item in pending:
        try:
            tts.synthesize_to_wav(item.text, item.out_path, speaker=item.speaker)
            _append_have_barked_id(HAVE_BARKED_CSV, item.image_id)
            already.add(item.image_id)
            written.append(item.out_path)
            print(item.out_path)
        except Exception as e:
            print(f"  Failed image_id={item.image_id} speaker={item.speaker}: "
                  f"{type(e).__name__}: {e}")

    return len(written), written


# ── Argparser ─────────────────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch-generate WAV files using Coqui VCTK-VITS TTS."
    )
    p.add_argument(
        "--text-field", default="description",
        help="CSV column name containing text to synthesize (default: description).",
    )
    p.add_argument(
        "--image-id-field", default="image_id",
        help="CSV column name for image_id (default: image_id).",
    )
    p.add_argument(
        "--device", default=None,
        help="Force device (cuda/cpu). Defaults to auto-detect.",
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help=(
            "Items to accumulate before flushing progress log (default: 32). "
            "VITS processes items individually so this controls log frequency only."
        ),
    )
    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_argparser().parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    already = _load_have_barked_ids(HAVE_BARKED_CSV)
    print(f"Loaded {len(already)} already-processed image_ids from have_barked.csv")

    input_csvs = _collect_input_csvs(IN_CSV_DIR)
    print(f"Found {len(input_csvs)} input CSV(s): "
          f"{[os.path.basename(p) for p in input_csvs]}")

    print("Pre-scanning CSVs …")
    total_rows, total_in_topic_fit = _prescan_csvs(input_csvs, args.image_id_field)
    pct = (total_in_topic_fit / total_rows * 100.0) if total_rows else 0.0
    print(f"  total_rows={total_rows}  "
          f"in_topic_fit={total_in_topic_fit} ({pct:.1f}%)")

    start_time = time.time()
    tts = CoquiVITS.load(device=args.device)

    successes = 0
    skipped_already = 0
    skipped_topic_fit = 0
    done = False
    pending: list[_PendingItem] = []

    def flush() -> None:
        nonlocal successes
        n, _ = _flush_batch(tts, pending, already)
        successes += n
        pending.clear()

    def _log_progress() -> None:
        rows_touched = successes + skipped_already + skipped_topic_fit
        pct_rows = (rows_touched / total_rows * 100.0) if total_rows else 0.0
        topic_done = successes + skipped_already
        pct_topic = (topic_done / total_in_topic_fit * 100.0) if total_in_topic_fit else 0.0
        elapsed = time.time() - start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        rate = successes / elapsed if elapsed > 0 else 0.0
        print(
            f"[{h:02d}:{m:02d}:{s:02d}]",
            "Progress:",
            f"processed={successes} ({rate:.2f}/s)",
            f"skipped_already={skipped_already}",
            f"skipped_topic_fit={skipped_topic_fit}",
            f"rows_touched={rows_touched}/{total_rows} ({pct_rows:.1f}%)",
            f"topic_fit_range=[{TOPIC_FIT_MIN},{TOPIC_FIT_MAX})",
            f"done_of_topic_fit={topic_done}/{total_in_topic_fit} ({pct_topic:.1f}%)",
        )

    for input_csv in input_csvs:
        if done:
            break
        print(f"\n--- Processing {os.path.basename(input_csv)} ---")
        for row in _iter_rows(input_csv):
            image_id = _safe_int(row.get(args.image_id_field))
            if image_id is None:
                continue
            if image_id in already:
                skipped_already += 1
                continue

            fit_raw = row.get(TOPIC_FIT_FIELD)
            try:
                fit = (
                    float(fit_raw)
                    if fit_raw is not None and str(fit_raw).strip() != ""
                    else None
                )
            except Exception:
                fit = None
            if fit is None or fit < TOPIC_FIT_MIN or fit >= TOPIC_FIT_MAX:
                skipped_topic_fit += 1
                continue

            text = str(row.get(args.text_field, "")).strip()
            if not text:
                continue

            speaker = tts.random_speaker()
            out_path = _build_out_path(OUT_DIR, image_id=image_id, speaker=speaker)
            pending.append(_PendingItem(
                image_id=image_id, text=text,
                out_path=out_path, speaker=speaker,
            ))

            if len(pending) >= args.batch_size:
                flush()
                if successes % 100 == 0 and successes > 0:
                    _log_progress()

            if MAX_PROCESSED and successes >= MAX_PROCESSED:
                done = True
                break

    if pending and not done:
        flush()

    _log_progress()

    elapsed = time.time() - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    rate = successes / elapsed if elapsed > 0 else 0.0
    print(f"\n[{h:02d}:{m:02d}:{s:02d}] Final: "
          f"processed={successes} ({rate:.2f}/s)  output_dir={OUT_DIR}")


if __name__ == "__main__":
    main()
