import pandas as pd
import os
import soundfile as sf
import numpy as np

# Read all rows from the CSV file
df = pd.read_csv("barks/output_file.csv")

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
FADEOUT = 4

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

# Iterate through each row in the CSV file
for _, row in df.iterrows():
    input_path = os.path.join("barks/", row['out_name'])

    # Read the audio file
    audio_data, sample_rate = sf.read(input_path)

    # Adjusting volume level and applying panning
    EXP = 2.5
    volume_fit = float(row['topic_fit'])  # Using topic_fit as the volume level
    linear_scaled_vol = (volume_fit - FIT_VOL_MIN) / (FIT_VOL_MAX  - FIT_VOL_MIN) * (VOLUME_MAX - VOLUME_MIN) + VOLUME_MIN
    exp_scaled_vol = (volume_fit - FIT_VOL_MIN)**EXP / (FIT_VOL_MAX  - FIT_VOL_MIN)**EXP * (VOLUME_MAX - VOLUME_MIN) + VOLUME_MIN
              
    pan = float(row['pan'])  # Using pan as the panning level
    audio_data_adjusted = audio_data * exp_scaled_vol
    print("volume_fit:", volume_fit, "scaled_vol" ,exp_scaled_vol, "Pan:", pan)

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
max_amplitude = max(np.max(np.abs(mixed_audio[:, 0])), np.max(np.abs(mixed_audio[:, 1])))
mixed_audio /= max_amplitude

# Define the output path for the mixdown audio file
output_path = "multitrack_mixdown_offset.wav"

# Write the multitrack mixdown audio to the output file
sf.write(output_path, mixed_audio, sample_rate, format='wav')
