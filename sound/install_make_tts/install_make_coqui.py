"""
Coqui VITS TTS batch runner.

Reads all CSVs from the `input_csvs/` folder next to this script, generates
WAV files with Coqui TTS (VCTK multi-speaker VITS), and maintains a
`metas_audio.csv` log so already-processed image_ids are skipped across runs.

metas_audio.csv contains every column from the source CSV plus a `filename`
column recording the output WAV name.  It is shared with install_make_tts.py
(Bark) so both engines deduplicate against the same log.

Score range: 0.0 <= topic_fit < 0.7  (Fish covers 0.7 <= topic_fit <= 1.0)
Speaker:     random VCTK speaker picked per line (109 available)
Output:      tts_bark_out/
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional, Set

import scipy.io.wavfile
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
OUT_DIR         = os.path.join(_HERE, "tts_bark_out")
METAS_AUDIO_CSV = os.path.join(_HERE, "metas_audio.csv")

TOPIC_FIT_FIELD = "topic_fit"
TOPIC_FIT_MIN   = 0.0
TOPIC_FIT_MAX   = 0.7

MAX_PROCESSED = 0  # 0 = no limit

# Full VCTK speaker list — used as fallback if tts.speakers is unavailable
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


# ── CSV helpers ───────────────────────────────────────────────────────────────

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


def _load_done_ids(path: str) -> Set[int]:
    """Load already-processed image_ids from metas_audio.csv."""
    if not os.path.exists(path):
        return set()
    ids: Set[int] = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            v = _safe_int(row.get("image_id"))
            if v is not None:
                ids.add(v)
    return ids


def _existing_audio_by_id(folder: str) -> dict[int, str]:
    """Map image_id -> basename for wavs already in OUT_DIR (any speaker/voice)."""
    by_id: dict[int, str] = {}
    if not os.path.isdir(folder):
        return by_id
    audio_exts = {".wav", ".mp3", ".flac"}
    for _root, _dirs, files in os.walk(folder):
        for fname in files:
            if os.path.splitext(fname)[1].lower() not in audio_exts:
                continue
            image_id = _safe_int(fname.split("_")[0])
            if image_id is None:
                continue
            by_id[image_id] = fname
    return by_id


def _append_metas_audio(
    path: str,
    row: dict,
    filename: str,
    fieldnames: list[str],
) -> None:
    """Append one completed row (original CSV columns + filename) to metas_audio.csv."""
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        out = {k: row.get(k, "") for k in fieldnames}
        out["filename"] = filename
        writer.writerow(out)


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


def _get_fieldnames(input_csvs: list[str]) -> list[str]:
    """Read the header row of the first input CSV and append 'filename'."""
    with open(input_csvs[0], "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # Force header read
        _ = reader.fieldnames
        base = list(reader.fieldnames) if reader.fieldnames else []
    if "filename" not in base:
        base.append("filename")
    return base


def _prescan_csvs(
    input_csvs: list[str], image_id_field: str
) -> tuple[int, int]:
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
    Wrapper around Coqui TTS VCTK-VITS with true GPU batch inference.

    Batch inference path uses the model internals directly so multiple texts
    are processed in a single forward pass.  Falls back to item-by-item
    synthesis if the internal API is unavailable (version guard).
    """
    _tts: object
    sample_rate: int
    speaker_list: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, device: Optional[str] = None) -> "CoquiVITS":
        from TTS.api import TTS

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading Coqui VCTK-VITS on {device} …")
        tts = TTS(
            model_name="tts_models/en/vctk/vits",
            progress_bar=False,
            gpu=(device == "cuda"),
        )

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

    # ── single-item fallback ──────────────────────────────────────────────────

    def synthesize_to_wav(self, text: str, out_wav_path: str, speaker: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(out_wav_path)) or ".", exist_ok=True)
        with open(os.devnull, "w") as _devnull, contextlib.redirect_stdout(_devnull):
            self._tts.tts_to_file(text=text, speaker=speaker, file_path=out_wav_path)
        return out_wav_path

    # ── true GPU batch inference ──────────────────────────────────────────────

    def synthesize_batch_to_wavs(
        self,
        texts: list[str],
        speakers: list[str],
        out_paths: list[str],
    ) -> list[str]:
        """
        Run a single batched VITS forward pass for all items.

        Accesses tts.synthesizer.tts_model directly so that multiple texts are
        processed together on GPU.  Sequences are left-padded to the longest
        item.  Raises RuntimeError if the internal API is unavailable so that
        _flush_batch can fall back to item-by-item synthesis.
        """
        model = self._tts.synthesizer.tts_model
        if not hasattr(model, "inference") or not hasattr(model, "tokenizer"):
            raise RuntimeError("VITS model internals not accessible — fallback required")

        # Tokenise
        seqs = [
            torch.LongTensor(model.tokenizer.text_to_ids(t, language=None))
            for t in texts
        ]
        x_lengths = torch.LongTensor([s.shape[0] for s in seqs])
        max_len = int(x_lengths.max().item())
        x = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, : s.shape[0]] = s

        # Speaker IDs
        name_to_id = model.speaker_manager.name_to_id
        sid = torch.LongTensor([name_to_id[spk] for spk in speakers])

        dev = next(model.parameters()).device
        with torch.inference_mode():
            outputs = model.inference(
                x.to(dev),
                aux_input={
                    "x_lengths": x_lengths.to(dev),
                    "speaker_ids": sid.to(dev),
                },
            )

        # outputs["model_outputs"] shape: (batch, 1, T)
        wavs = outputs["model_outputs"]

        written: list[str] = []
        for wav_t, path in zip(wavs, out_paths):
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            wav_np = wav_t.squeeze().cpu().float().numpy()
            scipy.io.wavfile.write(path, rate=self.sample_rate, data=wav_np)
            written.append(path)
        return written

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
    row: dict   # full original CSV row for metas_audio.csv logging


# ── Flush batch ───────────────────────────────────────────────────────────────

def _flush_batch(
    tts: CoquiVITS,
    pending: list[_PendingItem],
    already: Set[int],
    fieldnames: list[str],
) -> tuple[int, list[str]]:
    if not pending:
        return 0, []

    texts    = [item.text     for item in pending]
    speakers = [item.speaker  for item in pending]
    paths    = [item.out_path for item in pending]

    # Attempt true GPU batch inference; fall back to item-by-item on any failure.
    try:
        written = tts.synthesize_batch_to_wavs(texts, speakers, paths)
    except Exception as e:
        print(f"  Batch inference failed ({type(e).__name__}: {e}) — falling back to sequential")
        written = []
        for item in pending:
            try:
                tts.synthesize_to_wav(item.text, item.out_path, speaker=item.speaker)
                written.append(item.out_path)
            except Exception as inner:
                print(f"    Failed image_id={item.image_id} speaker={item.speaker}: "
                      f"{type(inner).__name__}: {inner}")

    succeeded = 0
    written_set = set(written)
    for item in pending:
        if item.out_path in written_set:
            _append_metas_audio(
                METAS_AUDIO_CSV, item.row,
                os.path.basename(item.out_path), fieldnames,
            )
            already.add(item.image_id)
            succeeded += 1

    return succeeded, written


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
            "Number of texts per GPU forward pass (default: 32). "
            "Reduce if you get CUDA OOM on long descriptions."
        ),
    )
    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_argparser().parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    already = _load_done_ids(METAS_AUDIO_CSV)
    files_by_id = _existing_audio_by_id(OUT_DIR)
    print(f"Loaded {len(already)} already-processed image_ids from metas_audio.csv")
    print(f"Existing audio files in {OUT_DIR}: {len(files_by_id)}")

    input_csvs = _collect_input_csvs(IN_CSV_DIR)
    print(f"Found {len(input_csvs)} input CSV(s): "
          f"{[os.path.basename(p) for p in input_csvs]}")

    # Derive output fieldnames from the input CSV header + 'filename'
    fieldnames = _get_fieldnames(input_csvs)

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
        n, _ = _flush_batch(tts, pending, already, fieldnames)
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

            # Dedup check against metas_audio.csv log
            if image_id in already:
                skipped_already += 1
                continue

            # Topic-fit filter
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

            # WAV existence guard — any on-disk file for this image_id counts,
            # regardless of which random speaker was baked into the filename.
            existing_name = files_by_id.get(image_id)
            if existing_name:
                _append_metas_audio(
                    METAS_AUDIO_CSV, row, existing_name, fieldnames,
                )
                already.add(image_id)
                skipped_already += 1
                continue

            speaker  = tts.random_speaker()
            out_path = _build_out_path(OUT_DIR, image_id=image_id, speaker=speaker)

            pending.append(_PendingItem(
                image_id=image_id,
                text=text,
                out_path=out_path,
                speaker=speaker,
                row=row,
            ))

            if len(pending) >= args.batch_size:
                flush()
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
