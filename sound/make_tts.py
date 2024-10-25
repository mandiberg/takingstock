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

title = 'Please choose your operation: '
options = ['meta', 'bark', 'openai', 'eleven_labs']
OPTION, MODE = pick(options, title)

start = time.time()
io = DataIO()
INPUT = os.path.join(io.ROOTSSD, "audioproduction")
OUTPUT = os.path.join(io.ROOTSSD, "audioproduction/tts_files_test")
# Brandon paths
# INPUT = os.path.join(io.ROOTSSD, "sound")
# OUTPUT = os.path.join(io.ROOTSSD, "sound/tts_files_test")
WINDOW = [0,1]

TOPIC = 23
sourcefile = f"metas_{TOPIC}.csv"
output_csv = f"output_file_{TOPIC}.csv"

STOP_AFTER = 10000
counter = 1
start_at = 0

def get_existing_image_ids():
    existing_files = io.get_img_list(OUTPUT)
    existing_image_ids = [int(f.split("_")[0]) for f in existing_files if f.endswith(".wav")]
    return existing_image_ids

# Function to write TTS using Eleven Labs
def write_TTS_eleven_labs(client, input_text, file_name):
    # Select a random voice ID from the list
    voice_id = random.choice(VOICE_IDS) #found in API_11labs.py
    
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

    # Save the stream to an MP3 file
    with open(file_name, "wb") as f:
        for chunk in audio_stream:
            f.write(chunk)


def write_TTS_bark(input_text,file_name):
    inputs = processor(input_text, voice_preset=voice_preset)

    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    scipy.io.wavfile.write(file_name, rate=sample_rate, data=audio_array)

    return
    
def write_TTS_openai(input_text,file_name):
    voice_preset = random.choice(preset_list)
    response = client.audio.speech.create(
      model="tts-1",
      voice=voice_preset,
      input=input_text
    )
    ### SAMPLE RATE 24 kHz
    ## NO OPTION TO CHANGE THIS IN OPENAI
    ## BUT THERE ARE EXTERNAL LIBRARIES
    response.stream_to_file(file_name)
    return

def write_TTS_meta(input_text,file_name):
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        audio_array = model(**inputs).waveform
    audio_array = audio_array.cpu().numpy().squeeze()
    scipy.io.wavfile.write(file_name, rate=sample_rate, data=audio_array)
    #  https://huggingface.co/facebook/mms-tts-eng ##
    return

if OPTION=="openai":
    client = OpenAI(api_key=api_key) ##Michael's API key
    model="tts-1", ##(tts-1,tts-1-hd)
    #voice="alloy", ##(alloy, echo, fable, onyx, nova, and shimmer)
    # preset_list=["alloy", "echo", "fable", "onyx", "nova","shimmer","alloy", "fable", "nova","shimmer", "nova","shimmer", "nova","shimmer"] # this is a clunky way of prioritizing higher pitched voices
    preset_list=["alloy", "echo", "fable", "onyx", "nova","shimmer"]
    write_TTS=write_TTS_openai
    WINDOW = [.6,.75]
    
    
elif OPTION=="bark":
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark")
    sample_rate = model.generation_config.sample_rate
    preset_list = [f"v2/en_speaker_{i}" for i in range(10)]
    write_TTS=write_TTS_bark
    WINDOW = [.5,.6]
    
elif OPTION=="meta":
    model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    WINDOW = [0,.5]


    # processor = AutoProcessor.from_pretrained("suno/bark")
    # model = BarkModel.from_pretrained("suno/bark")
    # sample_rate = model.generation_config.sample_rate
    ## SINCE GARBAGE TO LOWER SAMPLE RATE   
    sample_rate=16000
    # preset_list = [f"v2/en_speaker_{i}" for i in range(10)]
    write_TTS=write_TTS_meta

elif OPTION == "eleven_labs":
    client = ElevenLabs(api_key=XI_API_KEY) #call the API key from API_11labs.py 
    write_TTS = write_TTS_eleven_labs
    WINDOW = [0.85, 1] #testing window I don't know what exactly I am looking for to determine the window so I set it to the same as OpenAI
    


with open(os.path.join(INPUT, sourcefile), mode='r',encoding='utf-8-sig', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    existing_image_ids = get_existing_image_ids()
    # Determine the mode for opening the output CSV file ('w' for new file, 'a' for append)
    mode = 'w' if not os.path.exists(os.path.join(OUTPUT, output_csv)) else 'a'

    # Open the output CSV file for writing (or appending)
    with open(os.path.join(OUTPUT, output_csv), mode, newline='') as output_csvfile:
        # Define the fieldnames for the output CSV file (including new column 'out_name')
        fieldnames = reader.fieldnames + ['out_name']
        writer = csv.DictWriter(output_csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate through each row
        for row in reader:
            print("row", row)
            image_id = int(row['image_id'])
            input_text = row['description']
            fit = float(row['topic_fit'])
            # print(f"-- {counter} -- Processing image {image_id} with text: {input_text}")
            if image_id in existing_image_ids:
                print(f"Skipping image_id {image_id} because it already exists {counter}")
                counter += 1
                continue
            # skip row until counter is greater than start_at
            elif counter < start_at:
                counter += 1
                continue
            elif input_text=="": 
                print(f"Skipping image_id {image_id} because it has no description {counter}")
                counter += 1
                continue
            elif fit<WINDOW[0] or fit>=WINDOW[1]:
                print(f"Skipping image_id {image_id} because {fit} doesn't fit in the window {WINDOW} -- {counter}")
                counter += 1
                continue
            if counter%10==0: print(counter,"sounds generated")                
            print(row)

            if OPTION == "eleven_labs":
                voice_id = random.choice(VOICE_IDS)  
                voice_index = VOICE_IDS.index(voice_id) + 1 
                out_name = f"{str(image_id)}_{OPTION}_v{voice_index}_{fit}.mp3"  # Save as MP3 for Eleven Labs
                write_TTS(client, input_text, os.path.join(OUTPUT, out_name)) 

            else:
                if OPTION != "meta":
                    voice_preset = random.choice(preset_list)
                    out_name = f"{str(image_id)}_{OPTION}_v{voice_preset[-1]}_{fit}.wav"
                else:
                    # No preset option for meta
                    out_name = f"{str(image_id)}_{OPTION}_{fit}.wav"  
                
                file_name = os.path.join(OUTPUT, out_name)
                write_TTS(input_text, file_name)
                # Write the row to the output CSV file with 'out_name' added

            row['out_name'] = out_name
            writer.writerow(row)

            counter += 1
            if counter > STOP_AFTER:
                break

print("Time:", time.time() - start)
