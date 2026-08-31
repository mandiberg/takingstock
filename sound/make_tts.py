from elevenlabs import ElevenLabs, VoiceSettings
import os
import csv
import random
import time
from API import api_key
from API_11labs import XI_API_KEY, VOICE_IDS
from openai import OpenAI
from pick import pick

# go get IO class from parent folder
# caution: path[0] is reserved for script path (or '' in REPL)
import sys
if sys.platform == "darwin": sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
# if sys.platform == "darwin": sys.path.insert(1, '/Users/brandonflores/Documents/gitHub/takingstock_brandon/')
elif sys.platform == "win32": sys.path.insert(1, 'C:/Users/jhash/Documents/GitHub/facemap2/')

if os.path.exists('/Users/tenchc/Documents/GitHub/takingstock/'):
    sys.path.insert(1, '/Users/tenchc/Documents/GitHub/takingstock/')

from mp_db_io import DataIO

# after you make_video, you to need put them in a folder and merge_expanded_images to produce a metas file.

title = 'Please choose your operation: '
options = ['meta', 'bark', 'openai_or_eleven_labs', 'fish']
OPTION, MODE = pick(options, title)

start = time.time()
io = DataIO()
print(io.ROOTSSD)
INPUT = os.path.join(io.ROOTSSD, "tts_sport")
OUTPUT = "/Volumes/OWC5/tts_sport"
# Brandon paths
# INPUT = os.path.join(io.ROOTSSD, "sound")
# OUTPUT = os.path.join(io.ROOTSSD, "sound/tts_files_test")
WINDOW = [0, 1]

TOPIC = 124
sourcefile = f"metas_{TOPIC}.csv"
METAS_AUDIO_CSV = os.path.join(OUTPUT, "metas_audio.csv")

STOP_AFTER = 10000000
counter = 1
start_at = 0

OPENAI_PRESET_LIST = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
OPENAI_VOICE_COUNT = len(OPENAI_PRESET_LIST)
ELEVEN_LABS_VOICE_COUNT = len(VOICE_IDS) if VOICE_IDS else 20
TOTAL_VOICES = OPENAI_VOICE_COUNT + ELEVEN_LABS_VOICE_COUNT
AUDIO_EXTS = {".wav", ".mp3", ".flac"}
FISH_MODEL = "s2.1-pro-free"
FISH_VOICE_PAGE_SIZE = 50
FISH_VOICE_MAX = 1000


def hashed_audio_path(out_dir, filename):
    """Two-level MD5 folders keyed on the full filename, including extension."""
    level1, level2 = io.get_hash_folders(filename)
    return os.path.join(out_dir, level1, level2, filename)


def _safe_int(value):
    try:
        if value is None:
            return None
        s = str(value).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


def load_metas_audio_ids(path):
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            v = _safe_int(row.get("image_id"))
            if v is not None:
                ids.add(v)
    return ids


def existing_audio_by_id(folder):
    """Walk hash folders (or a flat dir) and map image_id -> basename."""
    by_id = {}
    if not os.path.isdir(folder):
        return by_id
    for _root, _dirs, files in os.walk(folder):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in AUDIO_EXTS:
                continue
            image_id = _safe_int(fname.split("_")[0])
            if image_id is None:
                continue
            by_id[image_id] = fname
    return by_id


def metas_fieldnames(metas_path, source_fieldnames):
    if os.path.exists(metas_path) and os.path.getsize(metas_path) > 0:
        with open(metas_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            names = list(reader.fieldnames or [])
        if "filename" not in names:
            names.append("filename")
        return names
    names = list(source_fieldnames or [])
    if "filename" not in names:
        names.append("filename")
    return names


def append_metas_audio(path, row, filename, fieldnames):
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        out = {k: row.get(k, "") for k in fieldnames}
        out["filename"] = filename
        writer.writerow(out)


def write_TTS_eleven_labs(client, input_text, file_name, voice_id):
    audio_stream = client.text_to_speech.stream(
        voice_id=voice_id,
        output_format="mp3_22050_32",
        text=input_text,
        voice_settings=VoiceSettings(
            stability=0.1,
            similarity_boost=0.3,
            style=0.2,
        ),
    )

    os.makedirs(os.path.dirname(os.path.abspath(file_name)) or ".", exist_ok=True)
    with open(file_name, "wb") as f:
        for chunk in audio_stream:
            if chunk:
                f.write(chunk)


def write_TTS_bark(input_text, file_name):
    inputs = processor(input_text, voice_preset=voice_preset)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    os.makedirs(os.path.dirname(os.path.abspath(file_name)) or ".", exist_ok=True)
    scipy.io.wavfile.write(file_name, rate=sample_rate, data=audio_array)


def write_TTS_openai(client, input_text, file_name, voice_preset):
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice_preset,
        input=input_text,
        response_format="wav",
    )
    os.makedirs(os.path.dirname(os.path.abspath(file_name)) or ".", exist_ok=True)
    response.write_to_file(file_name)


def write_TTS_fish(client, input_text, file_name, voice_id):
    audio = client.tts.convert(
        text=input_text,
        reference_id=voice_id,
        format="wav",
        model=FISH_MODEL,
    )
    os.makedirs(os.path.dirname(os.path.abspath(file_name)) or ".", exist_ok=True)
    with open(file_name, "wb") as f:
        f.write(audio)


def load_fish_voice_ids(client):
    """Page through the public English voice library for as much variety as the API returns."""
    ids = []
    seen = set()
    page_number = 1
    while len(ids) < FISH_VOICE_MAX:
        page = client.voices.list(
            page_size=FISH_VOICE_PAGE_SIZE,
            page_number=page_number,
            self_only=False,
            language=["en"],
            sort_by="created_at",
        )
        items = list(getattr(page, "items", None) or [])
        if not items:
            break
        for voice in items:
            voice_id = getattr(voice, "id", None) or getattr(voice, "_id", None)
            if not voice_id or voice_id in seen:
                continue
            seen.add(voice_id)
            ids.append(voice_id)
        total = getattr(page, "total", None)
        print(f"  Fish voices page {page_number}: +{len(items)} (pool={len(ids)}"
              + (f", total={total}" if total is not None else "") + ")")
        if len(items) < FISH_VOICE_PAGE_SIZE:
            break
        if total is not None and page_number * FISH_VOICE_PAGE_SIZE >= total:
            break
        page_number += 1
    return ids


def write_TTS_meta(input_text, file_name):
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        audio_array = model(**inputs).waveform
    audio_array = audio_array.cpu().numpy().squeeze()
    os.makedirs(os.path.dirname(os.path.abspath(file_name)) or ".", exist_ok=True)
    scipy.io.wavfile.write(file_name, rate=sample_rate, data=audio_array)


def select_voice_and_client(api_key_openai, api_key_elevenlabs):
    voice_index = random.randint(1, TOTAL_VOICES)
    if voice_index <= OPENAI_VOICE_COUNT:
        client = OpenAI(api_key=api_key_openai)
        voice_preset = random.choice(OPENAI_PRESET_LIST)
        return client, write_TTS_openai, voice_preset, "wav"
    client = ElevenLabs(api_key=api_key_elevenlabs)
    voice_id = VOICE_IDS[voice_index - OPENAI_VOICE_COUNT - 1]
    return client, write_TTS_eleven_labs, voice_id, "mp3"


if OPTION == "openai_or_eleven_labs":
    WINDOW = [0.7, 1]

elif OPTION == "fish":
    from fishaudio import FishAudio
    from API_fish import FISH_API_KEY

    fish_client = FishAudio(api_key=FISH_API_KEY)
    print(f"Loading Fish public English voices ({FISH_MODEL}) …")
    FISH_VOICE_IDS = load_fish_voice_ids(fish_client)
    if not FISH_VOICE_IDS:
        raise SystemExit("Fish voice library returned no voices. Check API_fish.py and the API key.")
    print(f"Fish voice pool: {len(FISH_VOICE_IDS)}")
    WINDOW = [0.7, 1]

elif OPTION == "bark":
    from transformers import AutoProcessor, BarkModel
    import torch
    import scipy

    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Bark running on: {device}")
    sample_rate = model.generation_config.sample_rate
    preset_list = [f"v2/en_speaker_{i}" for i in range(10)]
    write_TTS = write_TTS_bark
    WINDOW = [.5, .7]

elif OPTION == "meta":
    from transformers import VitsModel, AutoTokenizer
    import torch
    import scipy

    model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    WINDOW = [0, .5]
    sample_rate = 16000
    write_TTS = write_TTS_meta

os.makedirs(OUTPUT, exist_ok=True)

already = load_metas_audio_ids(METAS_AUDIO_CSV)
files_by_id = existing_audio_by_id(OUTPUT)
print(f"Already in metas_audio.csv: {len(already)}")
print(f"Existing audio files: {len(files_by_id)}")

source_path = os.path.join(INPUT, sourcefile)
print(f"Source file: {source_path}")
with open(source_path, mode="r", encoding="utf-8-sig", newline="") as _f:
    source_reader = csv.DictReader(_f)
    source_fieldnames = list(source_reader.fieldnames or [])
    _lines_to_process = sum(
        1 for row in source_reader
        if _safe_int(row.get("image_id")) not in already
        and _safe_int(row.get("image_id")) not in files_by_id
        and row.get("description")
        and WINDOW[0] <= float(row["topic_fit"]) < WINDOW[1]
    )
print(f"Lines to process: {_lines_to_process} (window {WINDOW})")

fieldnames = metas_fieldnames(METAS_AUDIO_CSV, source_fieldnames)
processed = 0

with open(source_path, mode="r", encoding="utf-8-sig", newline="") as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        image_id = _safe_int(row.get("image_id"))
        input_text = row.get("description") or ""
        try:
            fit = float(row["topic_fit"])
        except (TypeError, ValueError, KeyError):
            counter += 1
            continue

        if image_id is None:
            counter += 1
            continue
        if image_id in already:
            print(f"~~~ {image_id} (in metas_audio.csv)")
            counter += 1
            continue
        if image_id in files_by_id:
            print(f"~~~ {image_id} (audio exists, logging to metas_audio.csv)")
            append_metas_audio(METAS_AUDIO_CSV, row, files_by_id[image_id], fieldnames)
            already.add(image_id)
            counter += 1
            continue
        if counter < start_at:
            counter += 1
            continue
        if not input_text:
            print(f"- {image_id} (no description)")
            counter += 1
            continue
        if fit < WINDOW[0] or fit >= WINDOW[1]:
            print(f"Skipping image_id {image_id} (fit {fit} outside window {WINDOW})")
            counter += 1
            continue

        processed += 1
        pct = (100.0 * processed / _lines_to_process) if _lines_to_process else 100.0
        print(f"{processed} of {_lines_to_process} rows processed ({pct:.1f}%)")

        try:
            if OPTION == "openai_or_eleven_labs":
                client, write_TTS, voice_id_or_preset, file_extension = select_voice_and_client(api_key, XI_API_KEY)
                engine = "openai" if file_extension == "wav" else "elevenlabs"
                out_name = f"{image_id}_{engine}_{voice_id_or_preset}_{fit}.{file_extension}"
                file_path = hashed_audio_path(OUTPUT, out_name)
                print(f"  ++++++++  doing {engine}", image_id, "->", os.path.join(*io.get_hash_folders(out_name), out_name))
                write_TTS(client, input_text, file_path, voice_id_or_preset)
            elif OPTION == "fish":
                voice_id = random.choice(FISH_VOICE_IDS)
                out_name = f"{image_id}_fish_{voice_id}_{fit}.wav"
                file_path = hashed_audio_path(OUTPUT, out_name)
                print(f"  ++++++++  doing fish", image_id, "->", os.path.join(*io.get_hash_folders(out_name), out_name))
                write_TTS_fish(fish_client, input_text, file_path, voice_id)
            else:
                if OPTION != "meta":
                    voice_preset = random.choice(preset_list)
                    out_name = f"{image_id}_{OPTION}_v{voice_preset[-1]}_{fit}.wav"
                else:
                    out_name = f"{image_id}_{OPTION}_{fit}.wav"
                file_path = hashed_audio_path(OUTPUT, out_name)
                write_TTS(input_text, file_path)
        except Exception as e:
            print(f"  FAILED {image_id}: {type(e).__name__}: {e}")
            counter += 1
            continue

        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"  FAILED {image_id}: empty output {file_path}")
            counter += 1
            continue

        append_metas_audio(METAS_AUDIO_CSV, row, out_name, fieldnames)
        already.add(image_id)

        counter += 1
        if counter > STOP_AFTER:
            break

print("Total processing time:", time.time() - start)
print("metas_audio.csv:", METAS_AUDIO_CSV)
