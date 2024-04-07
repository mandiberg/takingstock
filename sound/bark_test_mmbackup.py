from transformers import AutoProcessor, BarkModel
import scipy
import os
import csv 
import random
import time
start = time.time()

# conda activate bark

INPUT = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/segment_images/audioproduction/bark"
OUTPUT = INPUT
# OUTPUT = "/Users/michaelmandiberg/Documents/GitHub/facemap/sound/barks"
sourcefile = "metas.csv"
output_csv = "output_file.csv"
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
preset_list = [f"v2/en_speaker_{i}" for i in range(10)]
STOP_AFTER = 200
counter = 1
start_at = 0 # next to complete

with open(os.path.join(INPUT, sourcefile), mode='r',encoding='utf-8-sig', newline='') as csvfile:
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
            # skip row until counter is greater than start_at
            if counter < start_at:
                counter += 1
                continue
            input_text = row['description']
            image_id = row['image_id']
            print(f"-- {counter} -- Processing image {image_id} with text: {input_text}")
            
            voice_preset = random.choice(preset_list)
            out_name = f"{str(image_id)}_voice{voice_preset[-1]}_{row['topic_fit']}.wav"
            inputs = processor(input_text, voice_preset=voice_preset)

            audio_array = model.generate(**inputs)
            audio_array = audio_array.cpu().numpy().squeeze()

            sample_rate = model.generation_config.sample_rate
            scipy.io.wavfile.write(os.path.join(OUTPUT, out_name), rate=sample_rate, data=audio_array)

            # Write the row to the output CSV file with 'out_name' added
            row['out_name'] = out_name
            writer.writerow(row)

            counter += 1
            if counter > STOP_AFTER:
                break

print("Time:", time.time() - start)