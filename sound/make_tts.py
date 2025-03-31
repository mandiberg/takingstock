from elevenlabs import ElevenLabs, VoiceSettings
from transformers import AutoProcessor, BarkModel, VitsModel, AutoTokenizer
import torch
import scipy
import os
import csv 
import random
import time
from API import api_key
from API_11labs import XI_API_KEY, VOICE_IDS  
# XI_API_KEY, VOICE_IDS = None, None
from openai import OpenAI
from pick import pick

# go get IO class from parent folder
# caution: path[0] is reserved for script path (or '' in REPL)
import sys
if sys.platform == "darwin": sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
# if sys.platform == "darwin": sys.path.insert(1, '/Users/brandonflores/Documents/gitHub/takingstock_brandon/')
elif sys.platform == "win32": sys.path.insert(1, 'C:/Users/jhash/Documents/GitHub/facemap2/')
from mp_db_io import DataIO

# after you make_video, you to need put them in a folder and merge_expanded_images to produce a metas file. 

title = 'Please choose your operation: '
options = ['meta', 'bark', 'openai_or_eleven_labs']  # Combine OpenAI and ElevenLabs
OPTION, MODE = pick(options, title)

start = time.time()
io = DataIO()
INPUT = os.path.join(io.ROOTSSD, "audioproduction")
OUTPUT = os.path.join(io.ROOTSSD, "tts_files_test")
# Brandon paths
# INPUT = os.path.join(io.ROOTSSD, "sound")
# OUTPUT = os.path.join(io.ROOTSSD, "sound/tts_files_test")
WINDOW = [0,1]

TOPIC = 32
sourcefile = f"metas_{TOPIC}.csv"
output_csv = f"output_file_{TOPIC}.csv"

STOP_AFTER = 100000
counter = 1
start_at = 0

OPENAI_PRESET_LIST=["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
OPENAI_VOICE_COUNT = len(OPENAI_PRESET_LIST)
ELEVEN_LABS_VOICE_COUNT = 20
TOTAL_VOICES = OPENAI_VOICE_COUNT + ELEVEN_LABS_VOICE_COUNT


def get_existing_image_ids():
    existing_files = io.get_img_list(OUTPUT)
    existing_image_ids = [int(f.split("_")[0]) for f in existing_files if (f.endswith(".wav") or f.endswith(".mp3"))]
    return existing_image_ids

# Function to write TTS using Eleven Labs
def write_TTS_eleven_labs(client, input_text, file_name, voice_id):
    audio_stream = client.text_to_speech.convert_as_stream(
        voice_id=voice_id,
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=input_text,
        voice_settings=VoiceSettings(
            stability=0.1,
            similarity_boost=0.3,
            style=0.2,
        ),
    )

    with open(file_name, "wb") as f:
        for chunk in audio_stream:
            f.write(chunk)

def write_TTS_bark(input_text, file_name):
    inputs = processor(input_text, voice_preset=voice_preset)
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    scipy.io.wavfile.write(file_name, rate=sample_rate, data=audio_array)

def write_TTS_openai(input_text, file_name):
    voice_preset = random.choice(OPENAI_PRESET_LIST)
    response = client.audio.speech.create(
      model="tts-1",
      voice=voice_preset,
      input=input_text
    )
    response.stream_to_file(file_name)

def write_TTS_meta(input_text, file_name):
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        audio_array = model(**inputs).waveform
    audio_array = audio_array.cpu().numpy().squeeze()
    scipy.io.wavfile.write(file_name, rate=sample_rate, data=audio_array)

def select_voice_and_client(api_key_openai, api_key_elevenlabs):
    voice_index = random.randint(1, TOTAL_VOICES)
    if voice_index <= OPENAI_VOICE_COUNT:
        preset_list = OPENAI_PRESET_LIST
        client = OpenAI(api_key=api_key_openai)
        voice_preset = random.choice(preset_list)
        write_TTS = write_TTS_openai
        file_extension = "wav"
        return client, write_TTS, voice_preset, file_extension
    else:
        client = ElevenLabs(api_key=api_key_elevenlabs)
        voice_id = VOICE_IDS[voice_index - OPENAI_VOICE_COUNT - 1]
        write_TTS = write_TTS_eleven_labs
        file_extension = "mp3"
        return client, write_TTS, voice_id, file_extension

if OPTION == "openai_or_eleven_labs":
    WINDOW = [0.6, 1] 
    
    # if random.choice([True, False]):  # Randomly choose between OpenAI and ElevenLabs
    #     client = OpenAI(api_key=api_key)  
    #     model="tts-1"
    #     preset_list=["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    #     write_TTS = write_TTS_openai
    #     WINDOW = [.6, .75]
    # else:
    #     client = ElevenLabs(api_key=XI_API_KEY)  
    #     write_TTS = write_TTS_eleven_labs
    #     WINDOW = [0.85, 1] 

elif OPTION == "bark":
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark")
    sample_rate = model.generation_config.sample_rate
    preset_list = [f"v2/en_speaker_{i}" for i in range(10)]
    write_TTS = write_TTS_bark
    WINDOW = [.5, .6]

elif OPTION == "meta":
    model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    WINDOW = [0, .5]
    sample_rate = 16000
    write_TTS = write_TTS_meta

with open(os.path.join(INPUT, sourcefile), mode='r', encoding='utf-8-sig', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    existing_image_ids = get_existing_image_ids()
    mode = 'w' if not os.path.exists(os.path.join(OUTPUT, output_csv)) else 'a'

    with open(os.path.join(OUTPUT, output_csv), mode, newline='') as output_csvfile:
        fieldnames = reader.fieldnames + ['out_name']
        writer = csv.DictWriter(output_csvfile, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()

        for row in reader:
            image_id = int(row['image_id'])
            input_text = row['description']
            fit = float(row['topic_fit'])

            # Skip conditions
            if image_id in existing_image_ids:
                print(f"~~~ {image_id} (exists)")
                counter += 1
                continue
            elif counter < start_at:
                counter += 1
                continue
            elif not input_text:
                print(f"- {image_id} (no description)")
                counter += 1
                continue
            elif fit < WINDOW[0] or fit >= WINDOW[1]:
                print(f"Skipping image_id {image_id} (fit {fit} outside window {WINDOW})")
                counter += 1
                continue

            # if counter % 10 == 0: 
            print(f"{counter} rows processed")

            # Dynamic TTS selection
            if OPTION == "openai_or_eleven_labs":
                client, write_TTS, voice_id_or_preset, file_extension = select_voice_and_client(api_key, XI_API_KEY)
                
                # Generate output filename and perform TTS
                out_name = f"{image_id}_{'openai' if file_extension == 'wav' else 'elevenlabs'}_{voice_id_or_preset}_{fit}.{file_extension}"
                file_path = os.path.join(OUTPUT, out_name)
                
                if file_extension == "wav":
                    print("  ++++++++  doing openai", image_id)
                    write_TTS(input_text, file_path)
                else:
                    print("  ++++++++  doing elevenlabs", image_id)
                    write_TTS(client, input_text, file_path, voice_id_or_preset)
            else:
                if OPTION != "meta":
                    voice_preset = random.choice(preset_list)
                    out_name = f"{image_id}_{OPTION}_v{voice_preset[-1]}_{fit}.wav"
                else:
                    out_name = f"{image_id}_{OPTION}_{fit}.wav"
                
                file_path = os.path.join(OUTPUT, out_name)
                write_TTS(input_text, file_path)

            # Record output
            row['out_name'] = out_name
            writer.writerow(row)

            # Increment counter and check stop condition
            counter += 1
            if counter > STOP_AFTER:
                break

print("Total processing time:", time.time() - start)