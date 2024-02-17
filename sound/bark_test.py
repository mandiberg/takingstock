from transformers import AutoProcessor, BarkModel
import scipy
import os
import csv 
import random

# time this process
import time
start = time.time()

INPUT = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/takingstock_dropbox/topic10_feb15_withMetas"
sourcefile = "metas.csv"
OUTPUT = "/Users/michaelmandiberg/Documents/GitHub/facemap/sound/barks"
filename = "bark.wav"
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
preset_list = [f"v2/en_speaker_{i}" for i in range(10)]
STOP_AFTER = 2

# read the CSV file
# df = pd.read_csv("sound_files.csv")

counter = 0
# Open the CSV file
with open(os.path.join(INPUT,sourcefile), newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    
    # Iterate through each row
    for row in reader:
        input_text = row['description']
        out_name = f"barktest+{str(counter)}_+{row['topic_fit']}.wav"
        voice_preset = random.choice(preset_list)
        inputs = processor(input_text, voice_preset=voice_preset)

        audio_array = model.generate(**inputs)
        audio_array = audio_array.cpu().numpy().squeeze()

        sample_rate = model.generation_config.sample_rate
        scipy.io.wavfile.write(os.path.join(OUTPUT, out_name), rate=sample_rate, data=audio_array)
        counter += 1
        if counter > STOP_AFTER:
            break


# # iterate through each row and use the model to generate the audio file
# for _, row in df.iterrows():
#     input_text = row['text']
#     voice_preset = row['voice_preset']
#     inputs = processor(input_text, voice_preset=voice_preset)

#     audio_array = model.generate(**inputs)
#     audio_array = audio_array.cpu().numpy().squeeze()

#     sample_rate = model.generation_config.sample_rate
#     scipy.io.wavfile.write(os.path.join(OUTPUT,row['file_path']), rate=sample_rate, data=audio_array)

# voice_preset = "v2/en_speaker_5"

# inputs = processor("Hello, my dog is cute", voice_preset=voice_preset)

# audio_array = model.generate(**inputs)
# audio_array = audio_array.cpu().numpy().squeeze()

# sample_rate = model.generation_config.sample_rate
# scipy.io.wavfile.write(os.path.join(OUTPUT,filename), rate=sample_rate, data=audio_array)

print("Time:", time.time() - start)