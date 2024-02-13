import pandas as pd
from pydub import AudioSegment
from pydub.playback import play

# Read the CSV file
df = pd.read_csv("sound_files.csv")

# Iterate through each row and play the audio file
for _, row in df.iterrows():
    input_path = "sound_files/" + row['file_path']
    output_path = "sound_output/" + row['file_path']
    volume_drop = .5 - float(row['fit_to_model'])  # Using fit_to_model as the volume level
    pan = float(row['pan'])  # Using fit_to_model as the volume level
    print("Playing:", input_path, "at volume level:", volume_drop)
    sound = AudioSegment.from_wav(input_path)
    
    # Adjusting volume level
    # adjusted_sound = sound - (sound.dBFS - volume_level * sound.dBFS)
    adjusted_sound = sound - (95 * volume_drop + 18)
    adjusted_sound = adjusted_sound.pan(pan)  # Adjust pan

    # save the adjusted sound
    adjusted_sound.export(output_path, format="wav")
    # play(adjusted_sound)
