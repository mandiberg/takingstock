import pandas as pd
import os
import soundfile as sf
import numpy as np
import librosa
import gc

# go get IO class from parent folder
# caution: path[0] is reserved for script path (or '' in REPL)
import sys
if sys.platform == "darwin": sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
elif sys.platform == "win32": sys.path.insert(1, 'C:/Users/jhash/Documents/GitHub/facemap2/')
from mp_db_io import DataIO


TOPIC=32 # what folder are the files in?

CSV_FILE = f"metas_{TOPIC}.csv"
SOUND_FOLDER = "tts_files_test"

# TOPICFOLDER = "topic" + str(TOPIC)

# start = time.time()
######Michael's folders##########
io = DataIO()
INPUT = os.path.join(io.ROOTSSD, "audioproduction")
#################################

######Satyam's folders###########
# INPUT = "C:/Users/jhash/Documents/GitHub/facemap2/sound"
#################################

# Choose a file starting with a given string
# prefixed = [filename for filename in os.listdir('.') if filename.startswith("prefix")]

# Read all rows from the CSV file
df = pd.read_csv(os.path.join(INPUT,CSV_FILE))
# sound_files_dict_image_id = io.get_existing_image_ids_from_wavs(INPUT,full_path=True)

# # Initialize lists to store audio data for each channel
# left_channel_data = []
# right_channel_data = []

# Sampling rate for the mixdown
sample_rate = None
TARGET_SAMPLE_RATE = 24000

# Offset/delay between each sample (in seconds)
OFFSET = 0.1
TRACK_COUNT = len(df)

VOLUME_MIN = 0
VOLUME_MAX = .8
FIT_VOL_MIN = .3
FIT_VOL_MAX = 1
FADEOUT = 7
QUIET =.5
KEYS = {
    32: ["shock", "surpris", "mouth", "confus", "shade", "fear", "express", "cover", "open", "excit"],
    34: ["achiev", "scream", "excit", "shout", "celebr", "success", "express", "aggress", "fist", "frustrat"]
}
good_files = []

def apply_fadeout(audio, sample_rate, duration=3.0):
    # convert to audio indices (samples)
    length = int(duration*sample_rate)
    end = audio.shape[0]
    start = end - length

    # compute fade out curve
    # linear fade
    fade_curve = np.linspace(1.0, 0.0, length)
    # apply the curve
    audio[start:end] = audio[start:end] * fade_curve

def conform_sample_rate(audio_data, sample_rate):
    if sample_rate != TARGET_SAMPLE_RATE:
        # Resample the audio to 24000 Hz
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
    return audio_data, sample_rate

def scale_volume_exp(volume_fit, exponent=3):
    exp_vol = (volume_fit - FIT_VOL_MIN)**exponent / (FIT_VOL_MAX  - FIT_VOL_MIN)**exponent * (VOLUME_MAX - VOLUME_MIN) + VOLUME_MIN
    return exp_vol

def scale_volume_linear(volume_fit, min_out = VOLUME_MIN, max_out = VOLUME_MAX):
    linear_vol = (volume_fit - FIT_VOL_MIN) / (FIT_VOL_MAX  - FIT_VOL_MIN) * (max_out - min_out) + min_out
    return linear_vol

def scale_volume(row, cycler):
    volume_fit = float(row['topic_fit'])  # Using topic_fit as the volume level 
    if search_for_keys(row):
        vol = scale_volume_exp(volume_fit,2)*.8
        # vol =0
        FADEOUT = 7
    elif volume_fit < QUIET:
        # vol = scale_volume_exp(volume_fit, 3)
        vol = scale_volume_linear(volume_fit, 0,.01)*cycler[0]
        # vol = 0
        FADEOUT = 15
    else:
        vol = scale_volume_linear(volume_fit, .01,.025)*cycler[1]
        FADEOUT = 15
        # vol = 0
    return vol, FADEOUT

def search_for_keys(row):
    # search the first three words of the description for each key in KEYS
    # if any of the keys are found, set the volume to 1
    # if not, set the volume to 0.5
    if pd.isna(row['description']): return False

    found = False
    for key in KEYS[TOPIC]:
        for word in row['description'].lower().split(" ")[:5]:
            if key in word:
                print(" ---- ", key, "found in", word, row['description'])
                return True
                break
    if not found:
        print("No keys found in", row['description'])
    return found

existing_files = io.get_img_list(os.path.join(INPUT, SOUND_FOLDER))
# make a dict of existing files using the first part of filename (split on _) as the key
existing_files = {os.path.basename(f).split("_")[0]:f for f in existing_files}

print("Existing files:", len(existing_files))
# print("Existing files:", (existing_files))
print("Existing file 1:", existing_files.keys())

def process_audio_chunk(chunk_df, existing_files, input_folder, start_index, chunk_index):
    left_channel_data = []
    right_channel_data = []
    max_end_time = 0
    for i, row in chunk_df.iterrows():
# Iterate through each row in the CSV file
# for i, row in df.iterrows():
        # use i to create a sine wave
        sin = np.sin(i/60)
        cos = abs(np.cos(i/60))
        cycler = [sin,cos]
    
        # input_path = os.path.join(INPUT, row['out_name'])
        # input_path = row['out_name']

        # if os.path.exists(input_path):
        #     good_files.append(input_path)
        # elif 
        # print("Row:", row)
        image_id = row['image_id']
        description = row['description']
        # print("Image ID:", image_id)
        if pd.notna(description) and image_id in existing_files:
            input_file = existing_files.get(str(image_id))
            print("Using existing file:", input_file)
        elif image_id:
            input_file = np.random.choice(list(existing_files.values()))
            print("assigned random file:", input_file)
        else:
            print("No good files found")
            continue

        input_path = os.path.join(INPUT,SOUND_FOLDER,input_file)

        # Read the audio file
        audio_data, sample_rate = sf.read(input_path)
        # print("Audio data shape:", audio_data.shape, "Sample rate:", sample_rate)
        audio_data, sample_rate = conform_sample_rate(audio_data, sample_rate)
        # print("Audio data shape:", audio_data.shape, "Sample rate:", sample_rate)
        
        # search for keys in the description
        # found = search_for_keys(row)

        try:
            # pull data from topic fit
            volume_fit = float(row['topic_fit'])  # Using topic_fit as the volume level
        except Exception as e:
            print("Error getting volume fit:", e)
            if type(row['topic_fit']) == str: continue
            else: volume_fit = 0.5
        # # Adjusting volume level and applying panning

        # pan = float(row['pan'])  # Using pan as the panning level
        # set pan to random value between -1 and 1
        pan = np.random.uniform(-1, 1)

        # FADEOUT = len(row['description']) *.5
        volume_scale, FADEOUT = scale_volume(row, cycler)
        audio_data_adjusted = audio_data * volume_scale
        # print(f"volume_fit:", volume_fit, "scaled_vol" ,volume_scale, "Pan:", pan, FADEOUT)

        if (FADEOUT * sample_rate) > len(audio_data_adjusted):
            FADEOUT = len(audio_data_adjusted) / sample_rate
        # Apply fadeout to the audio data
        apply_fadeout(audio_data_adjusted, sample_rate, FADEOUT)

        # If the audio is mono, duplicate the channel for both left and right channels
        if len(audio_data_adjusted.shape) == 1:
            audio_data_adjusted = np.column_stack((audio_data_adjusted, audio_data_adjusted))

        # Apply panning to the audio data
        audio_data_adjusted[:, 0] *= (1 - pan)  # Left channel
        audio_data_adjusted[:, 1] *= pan  # Right channel

        # # Append audio data to respective lists
        # left_channel_data.append(audio_data_adjusted[:, 0])
        # right_channel_data.append(audio_data_adjusted[:, 1])

        # Calculate the start time for this audio clip
        start_time = (start_index + i) * OFFSET
        end_time = start_time + len(audio_data_adjusted) / TARGET_SAMPLE_RATE
        max_end_time = max(max_end_time, end_time)
        
        # Create arrays with the correct offset
        left_channel = np.zeros(int(np.ceil(end_time * TARGET_SAMPLE_RATE)))
        right_channel = np.zeros(int(np.ceil(end_time * TARGET_SAMPLE_RATE)))
        
        # Insert the audio data at the correct position
        start_sample = int(start_time * TARGET_SAMPLE_RATE)
        end_sample = min(start_sample + len(audio_data_adjusted), len(left_channel))
        
        left_channel[start_sample:end_sample] = audio_data_adjusted[:end_sample-start_sample, 0]
        right_channel[start_sample:end_sample] = audio_data_adjusted[:end_sample-start_sample, 1]
        
        left_channel_data.append(left_channel)
        right_channel_data.append(right_channel)
    
    # Mix the audio data for the chunk
    max_length = max(len(data) for data in left_channel_data + right_channel_data)
    mixed_audio = np.zeros((max_length, 2))
    
    for left_channel, right_channel in zip(left_channel_data, right_channel_data):
        mixed_audio[:len(left_channel), 0] += left_channel
        mixed_audio[:len(right_channel), 1] += right_channel
    
    # Clear memory
    del left_channel_data, right_channel_data
    gc.collect()
    
    # save the mixed audio to a file
    # output_file = os.path.join(INPUT, f"multitrack_mixdown_offset_{TOPIC}_{chunk_index}.wav")
    # sf.write(output_file, mixed_audio, TARGET_SAMPLE_RATE, format='wav')

    return mixed_audio, max_end_time

def main():
    io = DataIO()
    INPUT = os.path.join(io.ROOTSSD, "audioproduction")
    
    # Read the CSV file in chunks
    chunk_size = 500  # Adjust this value based on your system's capabilities
    chunks = pd.read_csv(os.path.join(INPUT, CSV_FILE), chunksize=chunk_size)
    
    existing_files = io.get_img_list(os.path.join(INPUT, SOUND_FOLDER))
    existing_files = {os.path.basename(f).split("_")[0]:f for f in existing_files}
    
    output_file = os.path.join(INPUT, f"multitrack_mixdown_offset_{TOPIC}.wav")
    
    combined_audio = None
    start_index = 0
    
    for chunk_index, chunk in enumerate(chunks):
        chunk_audio, chunk_end_time = process_audio_chunk(chunk, existing_files, INPUT, start_index, chunk_index)
        print("Chunk audio length/sample:", len(chunk_audio)/TARGET_SAMPLE_RATE, "Chunk end time:", chunk_end_time)
        if combined_audio is None:
            combined_audio = chunk_audio
            print(chunk_index, "Combined audio shape:", combined_audio.shape, "Chunk audio shape:", chunk_audio.shape)
        else:
            # chunk_audio has silene that is the same length as len combined_audio
            # IDK where this is coming from, but I am just going to remove it
            
            # Remove 50 seconds of silence from the beginning of chunk_audio
            non_silent_index = np.argmax(np.abs(chunk_audio) > 0)
            non_silent_index = int(np.floor(non_silent_index / 2))
            print("Non-silent index:", non_silent_index)
            print("combined_audio shape:", combined_audio.shape, "chunk_audio shape:", chunk_audio.shape)
            np.set_printoptions(threshold=100)
            print(chunk_audio[:non_silent_index])
            print(chunk_audio[non_silent_index:])
            chunk_audio_without_silence = chunk_audio[non_silent_index:]
            # remove_silence(chunk_audio, 50, TARGET_SAMPLE_RATE)

            # Combine the original combined_audio and the processed chunk_audio
            combined_audio = np.concatenate((combined_audio, chunk_audio_without_silence))


            # # Extend combined_audio if necessary
            # print(chunk_index, "Combined audio shape:", combined_audio.shape, "Chunk audio shape:", chunk_audio.shape)
            # if len(chunk_audio) > len(combined_audio):
            #     padding = np.zeros((len(chunk_audio) - len(combined_audio), 2))
            #     combined_audio = np.vstack((combined_audio, padding))
            
            # # Add chunk_audio to combined_audio
            # combined_audio[:len(chunk_audio)] += chunk_audio
        
        # start_index += len(chunk)
        
        # Clear memory
        del chunk_audio
        gc.collect()
    
    # Normalize the final audio to prevent clipping
    max_amplitude = np.max(np.abs(combined_audio))
    if max_amplitude > 1.0:
        combined_audio /= max_amplitude
    
    # Write the final output file
    sf.write(output_file, combined_audio, TARGET_SAMPLE_RATE, format='wav')

if __name__ == "__main__":
    main()