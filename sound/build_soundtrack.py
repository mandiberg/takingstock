import pandas as pd
import os
import soundfile as sf
import numpy as np

# go get IO class from parent folder
# caution: path[0] is reserved for script path (or '' in REPL)
import sys
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
from mp_db_io import DataIO

TOPIC=17 # what folder are the files in?
TOPICFOLDER = "topic" + str(TOPIC)

# start = time.time()
######Michael's folders##########
io = DataIO()
INPUT = os.path.join(io.ROOT, "audioproduction", TOPICFOLDER)
OUTPUT = os.path.join(io.ROOT, "audioproduction", TOPICFOLDER)
#################################

######Satyam's folders###########
# INPUT = "C:/Users/jhash/Documents/GitHub/facemap2/sound"
# OUTPUT = "C:/Users/jhash/Documents/GitHub/facemap2/sound/sound_files/OpenAI"
#################################

# Choose a file starting with a given string
# prefixed = [filename for filename in os.listdir('.') if filename.startswith("prefix")]

# Read all rows from the CSV file
df = pd.read_csv(os.path.join(INPUT,"output_file.csv"))

# Initialize lists to store audio data for each channel
left_channel_data = []
right_channel_data = []

# Sampling rate for the mixdown
sample_rate = None

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
    7: ["scream", "excit", "rais", "celebr", "amaz", "success", "victori", "crazi", "surpris", "mouth"],
    17: ["point", "finger", "show", "number", "pictur", "smile", "confid", "idea", "clock", "hold"]
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

existing_files = io.get_img_list(INPUT)
# make a dict of existing files using the first part of filename (split on _) as the key
existing_files = {os.path.basename(f).split("_")[0]:f for f in existing_files}

print("Existing files:", len(existing_files))
print("Existing file 1:", existing_files.keys())
# Iterate through each row in the CSV file
for i, row in df.iterrows():
    # use i to create a sine wave
    sin = np.sin(i/60)
    cos = abs(np.cos(i/60))
    cycler = [sin,cos]
 
    input_path = os.path.join(INPUT, row['out_name'])
    # input_path = row['out_name']

    if os.path.exists(input_path):
        good_files.append(input_path)
    elif existing_files:
        input_file = existing_files.get(row['image_id'])
        if input_file:
            print("Using existing file:", input_file)
            input_path = os.path.join(INPUT,input_file)
        elif good_files:
            input_path = good_files[np.random.randint(0,len(good_files))]
        else:
            print("No good files found")
            continue
    else:
        print("No good files found")
        continue

    # Read the audio file
    audio_data, sample_rate = sf.read(input_path)

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
    print(f"volume_fit:", volume_fit, "scaled_vol" ,volume_scale, "Pan:", pan, FADEOUT)

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

    # Append audio data to respective lists
    left_channel_data.append(audio_data_adjusted[:, 0])
    right_channel_data.append(audio_data_adjusted[:, 1])

# Calculate the maximum length of audio data arrays
max_length = max(len(data) for data in left_channel_data + right_channel_data)

# Pad the shorter audio data arrays with zeros to match the length of the longer one
left_channel_data = [np.pad(data, (0, max_length - len(data)), mode='constant') for data in left_channel_data]
right_channel_data = [np.pad(data, (0, max_length - len(data)), mode='constant') for data in right_channel_data]

offset_samples = int(OFFSET * sample_rate)

# Iterate through each row and apply offset to the audio data arrays
for i in range(len(left_channel_data)):
    total_offset = offset_samples * TRACK_COUNT
    left_pad_width = i * offset_samples
    right_pad_width = total_offset - left_pad_width

    left_length = len(left_channel_data[i])
    right_length = len(right_channel_data[i])
    
    # Pad the audio data arrays
    left_channel_data[i] = np.pad(left_channel_data[i], (left_pad_width, right_pad_width), mode='constant')
    right_channel_data[i] = np.pad(right_channel_data[i], (left_pad_width, right_pad_width), mode='constant')

# Calculate the maximum length of audio data arrays
max_padded_length = max(len(data) for data in left_channel_data + right_channel_data)

# Calculate the number of samples to offset
offset_samples = int(sample_rate * OFFSET)
# print("Offset samples:", offset_samples)

# Create arrays to store the mixdown
mixed_audio = np.zeros((max_padded_length, 2))

# Mix the audio data for each row
for left_channel, right_channel in zip(left_channel_data, right_channel_data):
    # print("Shapes - left channel:", left_channel.shape, "right channel:", right_channel.shape)
    mixed_audio[:, 0] += left_channel
    mixed_audio[:, 1] += right_channel

# Normalize the mixdown audio to prevent clipping
# max_amplitude = max(np.max(np.abs(mixed_audio[:, 0])), np.max(np.abs(mixed_audio[:, 1])))
# mixed_audio /= max_amplitude

# Define the output path for the mixdown audio file
output_path = "multitrack_mixdown_offset_"+str(TOPIC)+".wav"

# Write the multitrack mixdown audio to the output file
sf.write(output_path, mixed_audio, sample_rate, format='wav')
