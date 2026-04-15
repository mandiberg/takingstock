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
os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "true"
import csv
import random
from dataclasses import dataclass
from typing import Iterable, Optional, Set

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

        # Flash Attention 2 gives further speedup; fall back silently if not installed
        try:
            model = BarkModel.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
            )
            print("Flash Attention 2 active.")
        except (ValueError, ImportError):
            model = BarkModel.from_pretrained(model_id, torch_dtype=torch_dtype)

        processor = AutoProcessor.from_pretrained(model_id)
        model = model.to(device)

        sample_rate = int(model.generation_config.sample_rate)
        preset_list = [f"v2/en_speaker_{i}" for i in range(10)]
        return cls(processor=processor, model=model, sample_rate=sample_rate, preset_list=preset_list)

    def synthesize_to_wav(self, text: str, out_wav_path: str, voice_preset: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(out_wav_path)) or ".", exist_ok=True)

        inputs = self.processor(text, voice_preset=voice_preset)
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.inference_mode():
            audio = self.model.generate(**inputs)

        audio_array = audio.detach().cpu().numpy().squeeze()
        scipy.io.wavfile.write(out_wav_path, rate=self.sample_rate, data=audio_array)
        return out_wav_path


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
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    already = _load_have_barked_ids(HAVE_BARKED_CSV)

    input_csvs = _collect_input_csvs(IN_CSV_DIR)
    print(f"Found {len(input_csvs)} input CSV(s): {[os.path.basename(p) for p in input_csvs]}")

    tts = BarkTTS.load(model_id=args.model_id, device=args.device, half=not args.no_half)

    successes = 0
    skipped_already_barked = 0
    skipped_topic_fit = 0
    done = False
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
            try:
                tts.synthesize_to_wav(text, out_path, voice_preset=voice_preset)
            except Exception as e:
                print(f"Failed image_id={image_id}: {e}")
                continue

            _append_have_barked_id(HAVE_BARKED_CSV, image_id)
            already.add(image_id)
            successes += 1
            print(out_path)

            if successes % 10 == 0:
                total_skipped = skipped_already_barked + skipped_topic_fit
                denom = successes + total_skipped
                pct_skipped = (total_skipped / denom * 100.0) if denom else 0.0
                print(
                    "Progress:",
                    f"processed={successes}",
                    f"skipped_already={skipped_already_barked}",
                    f"skipped_topic_fit={skipped_topic_fit}",
                    f"skipped_total={total_skipped}",
                    f"pct_skipped={pct_skipped:.1f}%",
                )

            if MAX_PROCESSED and successes >= MAX_PROCESSED:
                done = True
                break

    total_skipped = skipped_already_barked + skipped_topic_fit
    denom = successes + total_skipped
    pct_skipped = (total_skipped / denom * 100.0) if denom else 0.0
    print(
        "Final:",
        f"processed={successes}",
        f"skipped_already={skipped_already_barked}",
        f"skipped_topic_fit={skipped_topic_fit}",
        f"skipped_total={total_skipped}",
        f"pct_skipped={pct_skipped:.1f}%",
    )


if __name__ == "__main__":
    main()