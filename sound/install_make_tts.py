"""
Temporary Bark-only TTS batch runner.

Reads an input CSV, generates WAV files with BarkModel, and maintains a simple
log of processed image_ids in a CSV called `have_barked.csv`.
"""

from __future__ import annotations

import argparse
import os
os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "true"
import csv
import random
import sys
import warnings
from dataclasses import dataclass
from typing import Iterable, Optional, Set

# Allow running as `python sound/install_make_tts.py` (repo root module imports).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from mp_db_io import DataIO  # type: ignore
except ModuleNotFoundError:
    DataIO = None  # type: ignore

import scipy.io.wavfile
import torch
from transformers import AutoProcessor, BarkModel


# ----------------------------
# Edit these paths as needed
# ----------------------------
if DataIO is not None:
    io = DataIO()
    _ROOT = io.ROOTSSD
else:
    # Fallback: store audio under the repo root if DataIO isn't available.
    _ROOT = _REPO_ROOT
if DataIO is not None:
    io = DataIO()
    _ROOT = os.path.join(os.path.dirname(_REPO_ROOT), "taking_stock_production")
else:
    # Fallback: store audio under the target production folder if DataIO isn't available.
    _ROOT = os.path.join(os.path.dirname(_REPO_ROOT), "taking_stock_production")
print("ROOT is ", _ROOT)

AUDIO_FOLDER = os.path.join(_ROOT, "install_audio")
IN_CSV = os.path.join(AUDIO_FOLDER, "metas_11_bup.csv")
OUT_DIR = os.path.join(AUDIO_FOLDER, "tts_bark_out")
HAVE_BARKED_CSV = os.path.join(AUDIO_FOLDER, "have_barked.csv")

# Bark/Transformers warning cleanup
# - We make generation_config avoid the "max_new_tokens + max_length" combo.
# - We silence a few noisy internal warnings from transformers/bark that do not
#   affect generation, but spam stdout.
BARK_MAX_LENGTH = 1024
warnings.filterwarnings(
    "ignore",
    message=r"Passing `generation_config` together with generation-related arguments=.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"The attention mask and the pad token id were not set\..*",
)
warnings.filterwarnings(
    "ignore",
    message=r"The attention mask is not set and cannot be inferred from input.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"Both `max_new_tokens` .* and `max_length`.*",
)

# Only generate audio when TOPIC_FIT_MIN <= topic_fit < TOPIC_FIT_MAX
TOPIC_FIT_FIELD = "topic_fit"
TOPIC_FIT_MIN = 0.5
TOPIC_FIT_MAX = 0.6

# 0 = process entire CSV, otherwise stop after N successful generations
MAX_PROCESSED = 10


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
    def load(cls, model_id: str = "suno/bark", device: Optional[str] = None) -> "BarkTTS":
        processor = AutoProcessor.from_pretrained(model_id)
        model = BarkModel.from_pretrained(model_id)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # Avoid "Both max_new_tokens and max_length are set" warnings.
        # Transformers' GenerationConfig defaults max_length=20; setting None can
        # be normalized back to the default. Instead, disable max_new_tokens and
        # use a single explicit max_length.
        try:
            model.generation_config.max_new_tokens = None
            model.generation_config.max_length = int(BARK_MAX_LENGTH)
        except Exception:
            pass

        sample_rate = int(model.generation_config.sample_rate)
        preset_list = [f"v2/en_speaker_{i}" for i in range(10)]
        return cls(processor=processor, model=model, sample_rate=sample_rate, preset_list=preset_list)

    def synthesize_to_wav(self, text: str, out_wav_path: str, voice_preset: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(out_wav_path)) or ".", exist_ok=True)

        # Ensure generation_config stays consistent even if something mutates it.
        try:
            self.model.generation_config.max_new_tokens = None
            self.model.generation_config.max_length = int(BARK_MAX_LENGTH)
        except Exception:
            pass

        inputs = self.processor(text, voice_preset=voice_preset)
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.no_grad():
            audio = self.model.generate(**inputs)

        audio_array = audio.detach().cpu().numpy().squeeze()
        scipy.io.wavfile.write(out_wav_path, rate=self.sample_rate, data=audio_array)
        return out_wav_path


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
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    already = _load_have_barked_ids(HAVE_BARKED_CSV)

    tts = BarkTTS.load(model_id=args.model_id, device=args.device)

    successes = 0
    for row in _iter_rows(IN_CSV):
        image_id = _safe_int(row.get(args.image_id_field))
        if image_id is None:
            continue
        if image_id in already:
            continue

        fit_raw = row.get(TOPIC_FIT_FIELD)
        try:
            fit = float(fit_raw) if fit_raw is not None and str(fit_raw).strip() != "" else None
        except Exception:
            fit = None
        if fit is None or fit < TOPIC_FIT_MIN or fit >= TOPIC_FIT_MAX:
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

        if MAX_PROCESSED and successes >= MAX_PROCESSED:
            break


if __name__ == "__main__":
    main()