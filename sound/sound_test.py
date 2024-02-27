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

# Iterate through each row in the CSV file
for _, row in df.iterrows():
    input_path = os.path.join("barks/", row['out_name'])
    print("Doing:", input_path)

    # Read the audio file
    audio_data, file_sample_rate = sf.read(input_path)

    # Adjusting volume level and applying panning
    volume_drop = .5 - float(row['topic_fit'])  # Using topic_fit as the volume level
    pan = float(row['pan'])  # Using pan as the panning level
    audio_data_adjusted = audio_data * (10 ** (volume_drop / 20))

    # If the audio is mono, duplicate the channel for both left and right channels
    if len(audio_data_adjusted.shape) == 1:
        audio_data_adjusted = np.column_stack((audio_data_adjusted, audio_data_adjusted))

    audio_data_adjusted[:, 0] *= (1 - pan)  # Left channel
    audio_data_adjusted[:, 1] *= pan  # Right channel

    # Store the sampling rate
    sample_rate = file_sample_rate
    print("Sample rate:", sample_rate)

    # Append audio data to respective lists
    left_channel_data.append(audio_data_adjusted[:, 0])
    right_channel_data.append(audio_data_adjusted[:, 1])

# Calculate the maximum length of audio data arrays
max_length = max(len(data) for data in left_channel_data + right_channel_data)

# Pad the shorter audio data arrays with zeros to match the length of the longer one
left_channel_data = [np.pad(data, (0, max_length - len(data)), mode='constant') for data in left_channel_data]
right_channel_data = [np.pad(data, (0, max_length - len(data)), mode='constant') for data in right_channel_data]
for data in left_channel_data: print(data.shape)

offset_samples = int(OFFSET * sample_rate)

# Iterate through each row and apply offset to the audio data arrays
for i in range(len(left_channel_data)):
    total_offset = offset_samples * TRACK_COUNT
    left_pad_width = i * offset_samples
    right_pad_width = total_offset - left_pad_width

    left_length = len(left_channel_data[i])
    right_length = len(right_channel_data[i])
    
    # # Calculate the actual offset samples for this clip
    # actual_offset_samples = min(max_offset_samples, left_length)
    
    # # Calculate the padding widths
    # left_pad_width = max(actual_offset_samples, 0)
    # right_pad_width = max(max_length - left_length - actual_offset_samples, 0)
    print(i, "length", left_length, "Left pad width:", left_pad_width, "Right pad width:", right_pad_width)

    # Pad the audio data arrays
    left_channel_data[i] = np.pad(left_channel_data[i], (left_pad_width, right_pad_width), mode='constant')
    right_channel_data[i] = np.pad(right_channel_data[i], (left_pad_width, right_pad_width), mode='constant')

# Calculate the maximum length of audio data arrays
max_padded_length = max(len(data) for data in left_channel_data + right_channel_data)

# Calculate the number of samples to offset
offset_samples = int(sample_rate * OFFSET)
print("Offset samples:", offset_samples)

# Create arrays to store the mixdown
# mixed_audio = np.zeros((max_length + (len(left_channel_data) - 1) * offset_samples, 2))
mixed_audio = np.zeros((max_padded_length, 2))

print("Left channel data lengths:", [len(data) for data in left_channel_data])
print("Right channel data lengths:", [len(data) for data in right_channel_data])

# Mix the audio data for each row
for left_channel, right_channel in zip(left_channel_data, right_channel_data):
    print("Shapes - left channel:", left_channel.shape, "right channel:", right_channel.shape)
    mixed_audio[:, 0] += left_channel
    mixed_audio[:, 1] += right_channel

# Mix the audio data for each row with offset
# start_index = 0
# for left_channel, right_channel in zip(left_channel_data, right_channel_data):
#     end_index = start_index + len(left_channel)
#     # end_index = len(left_channel)
#     print("Start index:", start_index, "End index:", end_index)
#     print("Shapes - left channel:", left_channel.shape, "right channel:", right_channel.shape)
#     print("Length of left_channel:", len(left_channel))
#     print("Length of slice:", end_index - start_index)
    
#     print("Shape of mixed_audio slice:", mixed_audio[start_index:end_index, 0].shape)
#     print("Shape of left_channel:", left_channel.shape)

#     mixed_audio[start_index:end_index, 0] += left_channel
#     mixed_audio[start_index:end_index, 1] += right_channel
#     start_index = end_index + offset_samples


# Normalize the mixdown audio to prevent clipping
max_amplitude = max(np.max(np.abs(mixed_audio[:, 0])), np.max(np.abs(mixed_audio[:, 1])))
mixed_audio /= max_amplitude

# Define the output path for the mixdown audio file
output_path = "multitrack_mixdown_offset.wav"

# Write the multitrack mixdown audio to the output file
sf.write(output_path, mixed_audio, sample_rate, format='wav')
