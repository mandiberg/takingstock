import pandas as pd
from pydub import AudioSegment
from pydub.playback import play

# Read the CSV file
df = pd.read_csv("sound_files.csv")

# Iterate through each row and play the audio file
for _, row in df.iterrows():
    file_path = "sound_files/" + row['file_path']
    volume_drop = .5 - float(row['fit_to_model'])  # Using fit_to_model as the volume level
    pan = float(row['pan'])  # Using fit_to_model as the volume level
    print("Playing:", file_path, "at volume level:", volume_drop)
    sound = AudioSegment.from_wav(file_path)
    
    # Adjusting volume level
    # adjusted_sound = sound - (sound.dBFS - volume_level * sound.dBFS)
    adjusted_sound = sound - (100 * volume_drop)
    adjusted_sound = adjusted_sound.pan(pan)  # Adjust to the right

    play(adjusted_sound)
