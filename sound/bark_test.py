from transformers import AutoProcessor, BarkModel
import scipy
import os
import csv 
import random
import time
from API import api_key 
from openai import OpenAI

METHOD="openai" ##openai or bark


start = time.time()
######Michael's folders##########
# INPUT = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/takingstock_dropbox/topic10_feb15_withMetas"
# OUTPUT = "/Users/michaelmandiberg/Documents/GitHub/facemap/sound/barks"
#################################

######Satyam's folders###########
INPUT = "C:/Users/jhash/Documents/GitHub/facemap2/sound"
OUTPUT = "C:/Users/jhash/Documents/GitHub/facemap2/sound/sound_files/OpenAI"
#################################

sourcefile = "metas.csv"
output_csv = "output_file.csv"

STOP_AFTER = 100
counter = 1
start_at = 1

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
    response.stream_to_file(file_name)
    return

if METHOD=="openai":
    client = OpenAI(api_key=api_key) ##Michael's API key
    model="tts-1", ##(tts-1,tts-1-hd)
    #voice="alloy", ##(alloy, echo, fable, onyx, nova, and shimmer)
    preset_list=["alloy", "echo", "fable", "onyx", "nova","shimmer"]
    write_TTS=write_TTS_openai
    
elif METHOD=="bark":
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark")
    sample_rate = model.generation_config.sample_rate
    preset_list = [f"v2/en_speaker_{i}" for i in range(10)]
    write_TTS=write_TTS_bark
    


with open(os.path.join(INPUT, sourcefile), newline='') as csvfile:
    reader = csv.DictReader(csvfile)

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
            if counter%10==0: print(counter,"sounds generated")
            # skip row until counter is greater than start_at
            if counter < start_at:
                counter += 1
                continue
            input_text = row['description']
            voice_preset = random.choice(preset_list)
            out_name =METHOD+ f"test{str(counter)}_voice{voice_preset[-1]}_{row['topic_fit']}.wav"
            file_name=os.path.join(OUTPUT, out_name)
            write_TTS(input_text,file_name)
            # Write the row to the output CSV file with 'out_name' added
            row['out_name'] = out_name
            writer.writerow(row)

            counter += 1
            if counter > STOP_AFTER:
                break

print("Time:", time.time() - start)