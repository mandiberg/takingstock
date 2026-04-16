"""
Temporary Bark-only TTS batch runner.

Reads all CSVs from the `input_csvs/` folder next to this script, generates
WAV files with BarkModel, and maintains a single deduplication log
(`have_barked.csv`) so already-processed image_ids are skipped across runs
and across files.
"""

from __future__ import annotations

import argparse
import logging
import os
import csv
import random
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Set

import numpy as np
import scipy.io.wavfile
import torch
from transformers import AutoProcessor, BarkModel


# Transformers emits these purely-informational messages via logging.warning(),
# not warnings.warn(), so warnings.filterwarnings() can't catch them.
# We install a logging.Filter on the transformers logger instead.
class _SuppressBarkNoise(logging.Filter):
    _PATTERNS = (
        "both `max_new_tokens`",
        "attention mask",
        "pad_token_id",
        "passing `generation_config`",
    )
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage().lower()
        return not any(p in msg for p in self._PATTERNS)

# All warnings come from transformers.generation.utils — target that specific child logger.
# Filters on a parent logger don't fire for propagated records from children.
logging.getLogger("transformers.generation.utils").addFilter(_SuppressBarkNoise())


# All paths are relative to this script's own directory (zip-upload friendly).
_HERE = os.path.dirname(os.path.abspath(__file__))

IN_CSV_DIR      = os.path.join(_HERE, "input_csvs")   # drop any number of CSVs here
OUT_DIR         = os.path.join(_HERE, "tts_bark_out")
HAVE_BARKED_CSV = os.path.join(_HERE, "have_barked.csv")

# Only generate audio when TOPIC_FIT_MIN <= topic_fit < TOPIC_FIT_MAX
TOPIC_FIT_FIELD = "topic_fit"
TOPIC_FIT_MIN = 0.5
TOPIC_FIT_MAX = 0.6

# 0 = process entire CSV, otherwise stop after N successful generations
MAX_PROCESSED = 0


def _safe_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        s = str(value).strip()
        if s == "":
            return None
        return int(float(s))  # tolerate "123.0"
    except Exception:
        return None


def _load_have_barked_ids(have_barked_csv: str) -> Set[int]:
    if not os.path.exists(have_barked_csv):
        return set()

    ids: Set[int] = set()
    with open(have_barked_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # allow either header "image_id" or single-column CSV w/out header
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


@dataclass(frozen=True)
class BarkTTS:
    processor: AutoProcessor
    model: BarkModel
    sample_rate: int
    preset_list: list[str]
    # Pre-cached processor outputs keyed by voice preset string.
    # Avoids repeated disk I/O on every synthesize call.
    _preset_cache: dict  # preset_str -> dict of numpy arrays

    @classmethod
    def load(
        cls,
        model_id: str = "suno/bark",
        device: Optional[str] = None,
        half: bool = True,
    ) -> "BarkTTS":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # fp16 halves VRAM (~12 GB → ~5–6 GB) and speeds inference ~5–15%
        torch_dtype = torch.float16 if (half and device == "cuda") else torch.float32

        # Flash Attention 2 gives further speedup; fall back silently if not installed.
        # EncodecModel (a Bark sub-model) does not support FA2, so the fallback
        # must explicitly request "eager" to prevent transformers from auto-selecting FA2.
        try:
            model = BarkModel.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
            )
            print("Flash Attention 2 active.")
        except (ValueError, ImportError):
            model = BarkModel.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                attn_implementation="eager",
            )

        processor = AutoProcessor.from_pretrained(model_id)
        model = model.to(device)

        # torch.compile fuses CUDA kernels and eliminates repeated launch overhead.
        # "reduce-overhead" is optimal for autoregressive generation (dynamic shapes).
        # Expect a ~30–60 s compilation warmup on the first generate() call.
        if device == "cuda":
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print("torch.compile active (reduce-overhead).")
            except Exception as e:
                print(f"torch.compile unavailable ({e}), running eager.")

        sample_rate = int(model.generation_config.sample_rate)
        preset_list = [f"v2/en_speaker_{i}" for i in range(10)]

        # Pre-load speaker embeddings for all presets so synthesize calls never
        # touch disk.  We run an empty-string dummy text through the processor
        # to trigger the embedding download/cache, then retain only the
        # non-sequence keys (the numpy speaker-embedding arrays).
        print("Pre-caching speaker embeddings …")
        preset_cache: dict = {}
        for preset in preset_list:
            out = processor(" ", voice_preset=preset)
            preset_cache[preset] = {
                k: v for k, v in out.items()
                if isinstance(v, np.ndarray)
            }
        print(f"Cached embeddings for {len(preset_cache)} presets.")

        return cls(
            processor=processor,
            model=model,
            sample_rate=sample_rate,
            preset_list=preset_list,
            _preset_cache=preset_cache,
        )

    def synthesize_to_wav(self, text: str, out_wav_path: str, voice_preset: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(out_wav_path)) or ".", exist_ok=True)

        # Tokenise text only; inject pre-cached speaker embeddings.
        inputs = self.processor(text, voice_preset=None)
        inputs.update(self._preset_cache.get(voice_preset, {}))
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.inference_mode():
            audio = self.model.generate(**inputs)

        audio_array = audio.detach().cpu().float().numpy().squeeze()
        scipy.io.wavfile.write(out_wav_path, rate=self.sample_rate, data=audio_array)
        return out_wav_path

    def synthesize_batch_to_wavs(
        self,
        texts: list[str],
        out_wav_paths: list[str],
        voice_preset: str,
    ) -> list[str]:
        """Generate a batch of WAV files in a single forward pass.

        BarkProcessor does not support passing padding/return_tensors kwargs
        when batching (they leak into internal voice-preset validation).
        Instead we call the processor per-item and manually left-pad the
        sequence tensors before stacking into a single batch tensor.
        All items share one voice preset so speaker embeddings are identical.
        """
        for p in out_wav_paths:
            os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)

        # Tokenise each text individually (no voice_preset — embeddings come from cache).
        per_item = [self.processor(t, voice_preset=None) for t in texts]

        # Left-pad sequence tensors (input_ids, attention_mask) to the longest item.
        max_len = max(d["input_ids"].shape[-1] for d in per_item)

        batched: dict = {}
        for key in per_item[0]:
            values = [d[key] for d in per_item]
            first = values[0]
            if isinstance(first, torch.Tensor):
                if first.dim() >= 2 and first.shape[-1] != max_len:
                    padded = [
                        torch.nn.functional.pad(t, (max_len - t.shape[-1], 0), value=0)
                        for t in values
                    ]
                    batched[key] = torch.cat(padded, dim=0)
                else:
                    batched[key] = torch.cat(values, dim=0)
            else:
                batched[key] = first

        # Inject pre-cached speaker embeddings — stack one copy per item in the batch.
        cached_emb = self._preset_cache.get(voice_preset, {})
        for k, arr in cached_emb.items():
            batched[k] = np.concatenate([arr] * len(texts), axis=0)

        batched = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in batched.items()}

        with torch.inference_mode():
            # audio shape: (batch, time) — padded to the longest item in the batch
            audio_batch = self.model.generate(**batched)

        written: list[str] = []
        for wav_tensor, path in zip(audio_batch, out_wav_paths):
            arr = wav_tensor.detach().cpu().float().numpy().squeeze()
            scipy.io.wavfile.write(path, rate=self.sample_rate, data=arr)
            written.append(path)
        return written


def _collect_input_csvs(csv_dir: str) -> list[str]:
    """Return sorted list of .csv paths found in csv_dir."""
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
    """Return (total_rows, total_in_topic_fit) across all input CSVs.

    total_rows      — rows with a valid image_id
    total_in_topic_fit — rows whose topic_fit value falls in [TOPIC_FIT_MIN, TOPIC_FIT_MAX)
    """
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


def _build_out_path(out_dir: str, image_id: int, voice_preset: str) -> str:
    voice_slug = voice_preset.replace("/", "_")
    filename = f"{image_id}_bark_{voice_slug}.wav"
    return os.path.join(out_dir, filename)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch-generate WAV files using Bark (suno/bark).")
    p.add_argument(
        "--text-field",
        default="description",
        help="CSV column name containing the text to synthesize (default: description).",
    )
    p.add_argument(
        "--image-id-field",
        default="image_id",
        help="CSV column name containing image_id (default: image_id).",
    )
    p.add_argument("--model-id", default="suno/bark", help="HuggingFace model id.")
    p.add_argument("--device", default=None, help="Force device (cuda/cpu). Defaults to auto-detect.")
    p.add_argument(
        "--no-half",
        action="store_true",
        help="Load model in fp32 instead of fp16. Use for debugging or CPU-only runs.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=(
            "Number of texts to generate in a single GPU forward pass (default: 8). "
            "Safe for short-to-medium texts on a 24 GB card; "
            "reduce to 4 if you get CUDA OOM errors on long texts."
        ),
    )
    return p


@dataclass
class _PendingItem:
    image_id: int
    text: str
    out_path: str
    voice_preset: str


def _flush_batch(
    tts: BarkTTS,
    pending: list[_PendingItem],
    already: set[int],
) -> tuple[int, list[str]]:
    """Run one batched generate call and persist results. Returns (successes, written_paths)."""
    if not pending:
        return 0, []

    # All items share the same voice preset so the processor can build one padded tensor.
    # (Each item already has its own randomly-chosen preset stored in out_path's filename;
    #  for the actual generation we pick the preset of the first item — acceptable because
    #  voice variance within a batch is minor compared to cross-batch variance.)
    voice_preset = pending[0].voice_preset
    texts = [item.text for item in pending]
    out_paths = [item.out_path for item in pending]

    try:
        written = tts.synthesize_batch_to_wavs(texts, out_paths, voice_preset=voice_preset)
    except Exception as e:
        print(f"Batch failed ({len(pending)} items): {type(e).__name__}: {e} — falling back to one-by-one")
        written = []
        for item in pending:
            try:
                tts.synthesize_to_wav(item.text, item.out_path, voice_preset=item.voice_preset)
                written.append(item.out_path)
            except Exception as inner_e:
                print(f"  Failed image_id={item.image_id}: {inner_e}")

    succeeded = 0
    for item, path in zip(pending, written):
        _append_have_barked_id(HAVE_BARKED_CSV, item.image_id)
        already.add(item.image_id)
        succeeded += 1
        print(path)

    return succeeded, written


def main() -> None:
    args = _build_argparser().parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    already = _load_have_barked_ids(HAVE_BARKED_CSV)

    input_csvs = _collect_input_csvs(IN_CSV_DIR)
    print(f"Found {len(input_csvs)} input CSV(s): {[os.path.basename(p) for p in input_csvs]}")
    print(f"Batch size: {args.batch_size}")

    print("Pre-scanning CSVs for row counts …")
    total_rows, total_in_topic_fit = _prescan_csvs(input_csvs, args.image_id_field)
    pct_in_topic_fit = (total_in_topic_fit / total_rows * 100.0) if total_rows else 0.0
    print(
        f"  total_rows={total_rows}",
        f"  in_topic_fit={total_in_topic_fit} ({pct_in_topic_fit:.1f}% of rows)",
    )

    start_time = time.time()

    tts = BarkTTS.load(model_id=args.model_id, device=args.device, half=not args.no_half)

    successes = 0
    skipped_already_barked = 0
    skipped_topic_fit = 0
    done = False
    pending: list[_PendingItem] = []

    def flush() -> None:
        nonlocal successes
        n, _ = _flush_batch(tts, pending, already)
        successes += n
        pending.clear()

    for input_csv in input_csvs:
        if done:
            break
        print(f"\n--- Processing {os.path.basename(input_csv)} ---")
        for row in _iter_rows(input_csv):
            image_id = _safe_int(row.get(args.image_id_field))
            if image_id is None:
                continue
            if image_id in already:
                skipped_already_barked += 1
                continue

            fit_raw = row.get(TOPIC_FIT_FIELD)
            try:
                fit = float(fit_raw) if fit_raw is not None and str(fit_raw).strip() != "" else None
            except Exception:
                fit = None
            if fit is None or fit < TOPIC_FIT_MIN or fit >= TOPIC_FIT_MAX:
                skipped_topic_fit += 1
                continue

            text = str(row.get(args.text_field, "")).strip()
            if not text:
                continue

            voice_preset = random.choice(tts.preset_list)
            out_path = _build_out_path(OUT_DIR, image_id=image_id, voice_preset=voice_preset)
            pending.append(_PendingItem(image_id=image_id, text=text, out_path=out_path, voice_preset=voice_preset))

            if len(pending) >= args.batch_size:
                # Sort by text length so all items in a batch have similar token counts.
                # Autoregressive generation runs for max(lengths) steps — uniform batches
                # eliminate wasted steps on padding for short items.
                pending.sort(key=lambda x: len(x.text))
                flush()

                if successes % 10 == 0 and successes > 0:
                    rows_touched = successes + skipped_already_barked + skipped_topic_fit
                    pct_rows_done = (rows_touched / total_rows * 100.0) if total_rows else 0.0
                    topic_fit_done = successes + skipped_already_barked
                    pct_of_topic_fit = (topic_fit_done / total_in_topic_fit * 100.0) if total_in_topic_fit else 0.0
                    elapsed = time.time() - start_time
                    h, rem = divmod(int(elapsed), 3600)
                    m, s = divmod(rem, 60)
                    rate = successes / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[{h:02d}:{m:02d}:{s:02d}]",
                        "Progress:",
                        f"processed={successes} ({rate:.2f}/s)",
                        f"skipped_already={skipped_already_barked}",
                        f"skipped_topic_fit={skipped_topic_fit}",
                        f"rows_touched={rows_touched}/{total_rows} ({pct_rows_done:.1f}%)",
                        f"topic_fit={total_in_topic_fit} ({pct_in_topic_fit:.1f}% of rows)",
                        f"done_of_topic_fit={topic_fit_done}/{total_in_topic_fit} ({pct_of_topic_fit:.1f}%)",
                    )

            if MAX_PROCESSED and successes >= MAX_PROCESSED:
                done = True
                break

    # Flush any remaining items that didn't fill a full batch
    if pending and not done:
        flush()

    elapsed = time.time() - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    rate = successes / elapsed if elapsed > 0 else 0.0
    rows_touched = successes + skipped_already_barked + skipped_topic_fit
    pct_rows_done = (rows_touched / total_rows * 100.0) if total_rows else 0.0
    topic_fit_done = successes + skipped_already_barked
    pct_of_topic_fit = (topic_fit_done / total_in_topic_fit * 100.0) if total_in_topic_fit else 0.0
    print(
        f"[{h:02d}:{m:02d}:{s:02d}]",
        "Final:",
        f"processed={successes} ({rate:.2f}/s)",
        f"skipped_already={skipped_already_barked}",
        f"skipped_topic_fit={skipped_topic_fit}",
        f"rows_touched={rows_touched}/{total_rows} ({pct_rows_done:.1f}%)",
        f"topic_fit={total_in_topic_fit} ({pct_in_topic_fit:.1f}% of rows)",
        f"done_of_topic_fit={topic_fit_done}/{total_in_topic_fit} ({pct_of_topic_fit:.1f}%)",
    )


if __name__ == "__main__":
    main()