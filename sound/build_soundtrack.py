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
# SOUND_FOLDER = "37_metas_hold_for_now"

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
CHUNK_SIZE = 500  # Adjust this value based on your system's capabilities


###########
# open_ai_sr=24000
# bark=24000
# 11_labs=22000
# meta=16000
##########
# Offset/delay between each sample (in seconds)
OFFSET = 0.1
TRACK_COUNT = len(df)
#### GENERALLY 3 words per second
WPS=4
VOLUME_MIN = 0
VOLUME_MAX = .8
FIT_VOL_MIN = .3
FIT_VOL_MAX = 1
FADEOUT = 7
FADE_TIME = 1
QUIET =.5
LOUD_ALOWED = 2
loud_counter = []
KEYS = {
    0: ["sport", "exercis", "activ", "athlet", "fit", "train", "workout", "lifestyl", "healthi", "yoga"],
    1: ["outsid", "think", "sceneri", "landscap", "calm", "contempl", "peac", "retir", "pension", "blur"],
    2: ["chef", "kitchen", "cook", "apron", "cut", "hard", "occup", "food", "restaur", "uniform"],
    3: ["mustach", "player", "competit", "number", "soccer", "classroom", "limb", "ginger", "count", "curios"],
    4: ["occup", "adolesc", "employ", "expertis", "wisdom", "squar", "world", "project", "intellig", "composit"],
    5: ["denim", "pant", "pocket", "convers", "secur", "sweatshirt", "danger", "timber", "knee", "pigtail"],
    6: ["citi", "urban", "travel", "journey", "street", "sole", "vacat", "walk", "outdoor", "trip"],
    7: ["light", "phenomenon", "pictur", "natur", "brick", "cheek", "glow", "neutral", "lamp", "illumin"],
    8: ["vintag", "retro", "banner", "classic", "poster", "even", "cotton", "logo", "candi", "gown"],
    9: ["makeup", "fashion", "glamour", "beauti", "model", "eleg", "hair", "hairstyl", "style", "sensual"],
    10: ["drink", "reclin", "alcohol", "refresh", "bottl", "unusu", "chocol", "wine", "bunni", "rabbit"],
    11: ["busi", "corpor", "execut", "success", "manag", "offic", "suit", "profession", "confid", "worker"],
    12: ["shoe", "attitud", "determin", "pride", "individu", "desir", "cross", "challeng", "club", "length"],
    13: ["shadow", "multiraci", "magic", "plastic", "surgeri", "develop", "silhouett", "author", "tech", "attack"],
    14: ["stop", "skateboard", "ecolog", "skate", "exot", "extrem", "illustr", "poverti", "forbid", "friday"],
    15: ["muscl", "romant", "shape", "valentin", "heart", "muscular", "lift", "chest", "athlet", "bicep"],
    16: ["food", "eat", "diet", "fruit", "fresh", "healthi", "meal", "breakfast", "kitchen", "sweet"],
    17: ["garden", "plant", "farm", "rural", "growth", "agricultur", "nose", "farmer", "harvest", "natur"],
    18: ["board", "tone", "negat", "headach", "solut", "ribbon", "ecstat", "decis", "choic", "hindu"],
    19: ["masculin", "conscious", "macho", "eyebrow", "ladi", "eyelash", "perspect", "temptat", "deadlin", "old-fashion"],
    20: ["medic", "doctor", "colleg", "hospit", "stethoscop", "health", "healthcar", "medicin", "clinic", "nurs"],
    21: ["educ", "studi", "book", "student", "elementari", "univers", "schoolgirl", "learn", "read", "childhood"],
    22: ["finger", "gestur", "point", "thumb", "show", "symbol", "hand", "sign", "emot", "express"],
    23: ["depress", "stress", "problem", "mood", "frustrat", "sad", "worri", "tire", "heel", "balloon"],
    24: ["winter", "autumn", "cold", "fall", "warm", "scarf", "season", "snow", "forest", "natur"],
    25: ["fashion", "beauti", "pose", "model", "eleg", "hair", "skirt", "dress", "style", "studio"],
    26: ["hope", "pray", "funki", "religion", "billboard", "charact", "religi", "boot", "prayer", "cultur"],
    27: ["flower", "bouquet", "golden", "bride", "fight", "box", "move", "glove", "filter", "wild"],
    28: ["costum", "tradit", "halloween", "arabian", "fantasi", "carniv", "dress", "cultur", "mysteri", "primari"],
    29: ["set", "nutrit", "appl", "vitamin", "choos", "garland", "peel", "start", "knitwear", "individu"],
    30: ["real", "loss", "swimsuit", "vietnames", "villag", "agent", "center", "measur", "fabric", "reject"],
    31: ["innoc", "small", "childhood", "cute", "sweet", "play", "newborn", "beauti", "face", "happi"],
    32: ["shock", "surpris", "mouth", "confus", "shade", "fear", "express", "cover", "open", "excit"],
    33: ["advertis", "engag", "length", "blank", "quarter", "jump", "copi", "inform", "size", "plank"],
    34: ["achiev", "scream", "excit", "shout", "celebr", "success", "express", "aggress", "fist", "frustrat"],
    35: ["skin", "clean", "care", "fresh", "treatment", "health", "beauti", "healthi", "clear", "perfect"],
    36: ["seat", "tie", "chair", "wooden", "floor", "barefoot", "wife", "door", "housewif", "wood"],
    37: ["franc", "money", "win", "strip", "ball", "credit", "financ", "card", "currenc", "cash"],
    38: ["headshot", "dream", "hair", "view", "candid", "real", "focus", "foreground", "look", "imagin"],
    39: ["structur", "floral", "humor", "pattern", "tongu", "rock", "gold", "welcom", "stick", "sound"],
    40: ["internet", "laptop", "technolog", "digit", "tablet", "onlin", "communic", "wireless", "connect", "busi"],
    41: ["friend", "protect", "mask", "covid-19", "virus", "epidem", "diseas", "beverag", "medic", "divers"],
    42: ["coffe", "drink", "break", "cafe", "aspir", "electron", "exhaust", "restaur", "north", "downtown"],
    43: ["shop", "custom", "sale", "buy", "retail", "store", "purchas", "contact", "shopahol", "consumer"],
    44: ["observ", "singl", "teeth", "inform", "express", "confid", "emot", "posit", "studio", "cheer"],
    45: ["spring", "natur", "summer", "beach", "outdoor", "park", "grass", "vacat", "beauti", "activ"],
    46: ["labor", "construct", "engin", "industri", "muslim", "helmet", "tool", "safeti", "worker", "architect"],
    47: ["shirt", "cloth", "jean", "fashion", "studio", "casual", "handsom", "model", "pose", "style"],
    48: ["seduct", "swim", "lingeri", "underwear", "pool", "simplic", "bikini", "water", "culinari", "automobil"],
    49: ["music", "listen", "headphon", "danc", "perform", "nerd", "dancer", "teacher", "entertain", "audio"],
    50: ["object", "blow", "cloud", "wind", "bubbl", "kiss", "disabl", "solitud", "shampoo", "soap"],
    51: ["facad", "individu", "figur", "save", "invest", "retir", "economi", "inform", "chic", "account"],
    52: ["interior", "home", "room", "domest", "hous", "indoor", "live", "relax", "comfort", "sofa"],
    53: ["near", "button", "window", "businesswear", "teamwork", "cocktail", "binocular", "smoke", "press", "colleagu"],
    54: ["action", "time", "applic", "tattoo", "neckti", "textur", "watch", "clock", "histor", "wheel"],
    55: ["free", "anim", "relationship", "friendship", "togeth", "girlfriend", "pet", "famili", "coupl", "flirt"],
    56: ["daughter", "sick", "servic", "packag", "overweight", "parent", "deliveri", "transport", "unhealthi", "order"],
    57: ["satisfact", "collar", "secretari", "star", "well-dress", "reflect", "straw", "vest", "orient", "memori"],
    58: ["parti", "bald", "birthday", "instrument", "faith", "groom", "celebr", "music", "christian", "musician"],
    59: ["christma", "celebr", "present", "gift", "holiday", "santa", "decor", "festiv", "winter", "decemb"],
    60: ["authent", "game", "scienc", "virtual", "placard", "help", "milk", "innov", "templat", "brutal"],
    61: ["level", "infant", "plain", "artist", "paint", "race", "set", "draw", "fold", "mix"],
    62: ["offer", "sexual", "ident", "stone", "contain", "actor", "breast", "rear", "partnership", "ancient"],
    63: ["phone", "mobil", "communic", "telephon", "technolog", "messag", "talk", "smart", "text", "wireless"]
}
good_files = []

def check_fade_length(fade_length, audio_data_adjusted, sample_rate=TARGET_SAMPLE_RATE):
    if (fade_length * sample_rate) > len(audio_data_adjusted):
        fade_length = (len(audio_data_adjusted) / sample_rate)/2
    return fade_length

def apply_fadeout(audio, sample_rate, duration=3.0):
    duration = check_fade_length(duration, audio, sample_rate)
    # convert to audio indices (samples)
    length = int(duration*sample_rate)
    end = audio.shape[0]
    start = end - length

    # new
    # fade_time = int(FADE_TIME*sample_rate)
    # print("fade_time",fade_time)
    # print("length",length)
    # if fade_time > length:
    #     fade_time = length
    # print("fade_time after testing",fade_time)
    # compute fade out curve
    # # linear fade
    # fade_curve = np.linspace(1.0, 0.0, fade_time)

    # # add zeros to the end of the fade curve
    # fade_curve = np.append(fade_curve, np.zeros(length - fade_time))
    # print("fade_curve",len(fade_curve))

    fade_curve = np.power(np.linspace(1.0, 0.0, length),2)
    print("fade_curve",(fade_curve))
    # old

    # apply the curve
    audio[start:end] = audio[start:end] * fade_curve



def apply_fadein(audio, sample_rate, duration=3.0):
    duration = check_fade_length(duration, audio, sample_rate)
    print("sample_rate",sample_rate)
    # convert to audio indices (samples)
    print("duration",duration)
    print("len(audio)/samplerate",len(audio)/sample_rate)
    length = int(duration*sample_rate)
    print("length",length)
    end = length
    start = 0

    # compute fade out curve
    # linear fade
    fade_curve = np.power(np.linspace(0.0, 1.0, length),2)
    print(len(fade_curve),"len(fade_curve)")
    print(len(audio[start:end]),"len(audio[start:end])")
    print(len(audio),"len(audio)")
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

def calculate_fades(key_index,desc_count, audio_data, sample_rate):
    fadein = 0
    fadeout = 15
    wps = desc_count/(len(audio_data)/sample_rate)
    if len(key_index)>0:
        if len(key_index)==1:
            start,end=key_index[0],key_index[0]
        else:
            start,end=key_index[0],key_index[-1]
        # vol = scale_volume_linear(volume_fit, .3,1)
        fadein =   start/wps
        fadeout = (desc_count-end-1)/wps 
    return fadein,fadeout
def scale_volume(row, cycler, audio_data, sample_rate):
    def is_bark_loud(row):
        # image_id = float(row['topic_fit'])  # Using topic_fit as the volume level 
        image_id = row['image_id']  # Using topic_fit as the volume level
        path = existing_files.get(str(image_id))
        # if path containts meta, return True
        if "bark_v5" in path: return True
        else: return False

    global loud_counter
    volume_fit = float(row['topic_fit'])  # Using topic_fit as the volume level 
    # defaults
    fadein = 0
    fadeout = 15

    # search_for_keys to see where the matching keys are
    key_index,desc_count=search_for_keys(row)
    if volume_fit < QUIET:
        # vol = scale_volume_exp(volume_fit, 3)
        vol = scale_volume_linear(volume_fit, 0,.1)*cycler[0]
        vol = .001
    elif len(key_index)>0:
        # if keys are found, set the volume and fade in out based on the keys found
        fadein,fadeout=calculate_fades(key_index,desc_count, audio_data, sample_rate)
        vol = scale_volume_exp(volume_fit,1)*1
        print(key_index)
        # start,end=key_index[0],key_index[-1]
        # vol =0
        # if vol < .5: vol = .001
        if vol < QUIET: 
            if vol > QUIET/2:
                # vol = vol - len(loud_counter)*.1
                if len(loud_counter) == 0:
                    vol = scale_volume_linear(volume_fit,.4 ,.8)
                else:
                    vol = vol / (len(loud_counter)*.5+1)
                if np.max(np.abs(audio_data)) > .8: vol = vol/3
                # if vol > .8: vol = .8
                vol = .001
            else:
                vol = (vol*.5) *cycler[1]
                vol = .001
        elif is_bark_loud(row):
            if np.max(np.abs(audio_data)) > QUIET: vol = vol/3
        # else: vol = .001
    else:
        vol = scale_volume_linear(volume_fit, .04,.08)*cycler[1]
        # if vol > .1: vol = .1
        # vol = vol*cycler[1]
        print("cylcerl vol",vol)
        vol = .001
    return vol, fadeout,fadein

# def search_for_keys(row):
#     # search the first three words of the description for each key in KEYS
#     # if any of the keys are found, set the volume to 1
#     # if not, set the volume to 0.5
#     if pd.isna(row['description']): return False

#     found = False
#     for key in KEYS[TOPIC]:
#         for word in row['description'].lower().split(" ")[:5]:
#             if key in word:
#                 print(" ---- ", key, "found in", word, row['description'])
#                 return True
#                 break
#     if not found:
#         print("No keys found in", row['description'])
#     return found

def search_for_keys(row):
    # search the first three words of the description for each key in KEYS
    # if any of the keys are found, set the volume to 1
    # if not, set the volume to 0.5
    if pd.isna(row['description']): return [],0

    # found = False
    found_list=[]
    desc_split=row['description'].lower().split(" ")
    desc_count=len(desc_split)
    for index,word in enumerate(desc_split):
        for key in KEYS[TOPIC]:
            if key in word:
                print(" ---- ", key, "found in", word, row['description'],row['image_id'])
                found_list.append(index)
                break
    if len(found_list)==0:
        print("No keys found in", row['description'],"for topic model",KEYS[TOPIC])
    return found_list,desc_count

def test_repeat(description, last_description):
    # if the first three words of the description are the same as the last description
    print("Description:", description)
    if pd.notna(description) and pd.notna(last_description):
        if " ".join(description.split()[:3]) == " ".join(last_description.split()[:3]):
            return 1, description
        else:
            return 0, description
    else:
        return 0, description

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
    global loud_counter
    last_description = ""
    for i, row in chunk_df.iterrows():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# Iterate through each row in the CSV file
# for i, row in df.iterrows():
        # use i to create a sine wave
        sin = np.sin(i/60)
        cos = abs(np.cos(i/60))
        cycler = [sin,cos]
        print("Cycler:", cycler)
    
        # input_path = os.path.join(INPUT, row['out_name'])
        # input_path = row['out_name']

        # if os.path.exists(input_path):
        #     good_files.append(input_path)
        # elif 
        print("Row:", row)
        image_id = row['image_id']
        description = row['description']
        # print("Image ID:", image_id)
        # if str(image_id) in existing_files.keys(): 
        #     print("^^^^^^^^ Image_id in existing files already ^^^^^^^^^^^^^^^",image_id)
        #     print("file path",existing_files.get(str(image_id)))
        # else:
        #     print(image_id,"^^^^^^ image_id not in existing files ^^^^^^^^^^^")
        #     print("existing files",existing_files)

        if pd.notna(description) and str(image_id) in existing_files.keys():
            input_file = existing_files.get(str(image_id))
            print("Using existing file:", input_file)
        elif pd.notna(description) and image_id:
            input_file = np.random.choice(list(existing_files.values()))
            if row['topic_fit'] < .6:
                print("unprocessed meta file")
            elif row['topic_fit'] > .75:
                print("unprocessed openai file")
        elif pd.isna(description) and image_id:
            input_file = np.random.choice(list(existing_files.values()))
            if row['topic_fit'] > QUIET:
                row['topic_fit'] = row['topic_fit']/2
            print(f"is NaN assigned random file: {input_file} and topic_fit halved to {row['topic_fit']}")
        else:
            print("No good files found")
            continue

        input_path = os.path.join(INPUT,SOUND_FOLDER,input_file)

        # Read the audio file
        audio_data, sample_rate = sf.read(input_path)
        print("length at start",len(audio_data))
        print("location",input_path)
        # print("Audio data shape:", audio_data.shape, "Sample rate:", sample_rate)
        audio_data, sample_rate = conform_sample_rate(audio_data, sample_rate)
        # print("Audio data shape:", audio_data.shape, "Sample rate:", sample_rate)
        
        # search for keys in the description
        # found = search_for_keys(row)

        # I don't think this is still in use
        # try:
        #     # pull data from topic fit
        #     volume_fit = float(row['topic_fit'])  # Using topic_fit as the volume level
        # except Exception as e:
        #     print("Error getting volume fit:", e)
        #     if type(row['topic_fit']) == str: continue
        #     else: volume_fit = 0.5
        # # # Adjusting volume level and applying panning

        # pan = float(row['pan'])  # Using pan as the panning level
        # set pan to random value between -1 and 1
        pan = np.random.uniform(-1, 1)

        # fadeout = len(row['description']) *.5
        volume_scale, fadeout,fadein = scale_volume(row, cycler, audio_data, sample_rate)
        audio_data_adjusted = audio_data * volume_scale
        # print(f"volume_fit:", volume_fit, "scaled_vol" ,volume_scale, "Pan:", pan, fadeout)

        # count the loud audio files
        # subtract OFFSET from each value in the loud counter
        # do this each loop, regardless of the volume
        loud_delay_duration = 0
        loud_offset = 0

        if loud_counter and len(loud_counter) > 0:
            loud_counter = [x - OFFSET for x in loud_counter]
            print("Loud counter:", len(loud_counter))
            print("Loud counter:", loud_counter)
            # if any value in the loud counter is less than 0, remove it
            loud_counter = [x for x in loud_counter if x > 0]
        print("Loud counter:", len(loud_counter))
        if loud_counter and len(loud_counter) >= LOUD_ALOWED and volume_scale > QUIET:
            # audio_data_adjusted = audio_data_adjusted* (1/len(loud_counter))


            if len(loud_counter) > LOUD_ALOWED*4:
                    # if there is a backlog of loud files, reduce the volume and play normal speed
                    # otherwise, the track will be 3x long, with the last 2x all the loud files
                audio_data_adjusted = audio_data_adjusted* (6/(6+len(loud_counter)))
            else:
                print("TOO LOUD")
                loud_delay_duration = 2* (len(loud_counter)-LOUD_ALOWED)
                loud_offset = (max(loud_counter)/OFFSET) +(loud_delay_duration/OFFSET)
                print("Loud offset:", loud_offset)
        if volume_scale > QUIET:
            loud_counter.append(len(audio_data)/sample_rate)
            
        # Apply fadeout to the audio data
        apply_fadeout(audio_data_adjusted, sample_rate, fadeout)
        ################
        # Apply fadein to the audio data
        if fadein>0:apply_fadein(audio_data_adjusted, sample_rate, fadein)
        ####################
        # If the audio is mono, duplicate the channel for both left and right channels
        if len(audio_data_adjusted.shape) == 1:
            audio_data_adjusted = np.column_stack((audio_data_adjusted, audio_data_adjusted))

        # Apply panning to the audio data
        audio_data_adjusted[:, 0] *= (1 - pan)  # Left channel
        audio_data_adjusted[:, 1] *= pan  # Right channel

        # # Append audio data to respective lists
        # left_channel_data.append(audio_data_adjusted[:, 0])
        # right_channel_data.append(audio_data_adjusted[:, 1])
        repeat, last_description = test_repeat(description, last_description)
        # Calculate the start time for this audio clip
        # if repeat, then start at the same time as the last clip
        start_time = (start_index + i - repeat + loud_offset) * OFFSET
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

def merge_audio(combined_audio, chunk_audio_without_silence):
    # Assuming sample_rate is defined
    # sample_rate = TARGET_SAMPLE_RATE  # Example sample rate, replace with your actual sample rate
    overlap_duration = 10  # Duration in seconds
    overlap_samples = TARGET_SAMPLE_RATE * overlap_duration

    # Extract the last 10 seconds of combined_audio
    combined_audio_last_10s = combined_audio[-overlap_samples:]

    # Extract the first 10 seconds of chunk_audio_without_silence
    chunk_audio_first_10s = chunk_audio_without_silence[:overlap_samples]

    # Ensure both segments are the same length by padding the shorter one with zeros
    if len(combined_audio_last_10s) < overlap_samples:
        combined_audio_last_10s = np.pad(combined_audio_last_10s, (0, overlap_samples - len(combined_audio_last_10s)), 'constant')

    if len(chunk_audio_first_10s) < overlap_samples:
        chunk_audio_first_10s = np.pad(chunk_audio_first_10s, (0, overlap_samples - len(chunk_audio_first_10s)), 'constant')

    # Mix the audio by adding the arrays together
    overlapped_segment = combined_audio_last_10s + chunk_audio_first_10s

    # Concatenate the mixed segment with the remaining parts of combined_audio and chunk_audio_without_silence
    combined_audio = np.concatenate((combined_audio[:-overlap_samples], overlapped_segment, chunk_audio_without_silence[overlap_samples:]))
    # sf.write(str(len(c ombined_audio))+"combined_audio.wav", combined_audio, TARGET_SAMPLE_RATE, format='wav')
    return combined_audio

def main():
    io = DataIO()
    INPUT = os.path.join(io.ROOTSSD, "audioproduction")
    
    # Read the CSV file in chunks
    chunks = pd.read_csv(os.path.join(INPUT, CSV_FILE), chunksize=CHUNK_SIZE)
    
    existing_files = io.get_img_list(os.path.join(INPUT, SOUND_FOLDER))
    existing_files = {os.path.basename(f).split("_")[0]:f for f in existing_files}

    # get the intersection of the existing files and the image ids in the csv
    existing_files = {k: v for k, v in existing_files.items() if int(k) in df['image_id'].values}
    print("Existing files after INTERSECT:", len(existing_files))
    print("Existing file 1:", existing_files.keys())


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
            
            # find the point where the audio is no longer silent
            # non_silent_index_raw = np.argmax(np.abs(chunk_audio) > 0)
            non_silent_index_raw = np.argmax(np.abs(chunk_audio) > 0)
            # divide that by 2, because for some reason the line above returns 2x value
            non_silent_index = int(np.floor(non_silent_index_raw / 2))

            print("Non-silent index:", non_silent_index)
            print("combined_audio shape:", combined_audio.shape, "chunk_audio shape:", chunk_audio.shape)
            np.set_printoptions(threshold=100)
            print(chunk_audio[:non_silent_index])
            print(chunk_audio[non_silent_index:])
            chunk_audio_without_silence = chunk_audio[non_silent_index:]
            # remove_silence(chunk_audio, 50, TARGET_SAMPLE_RATE)

            # Combine the original combined_audio and the processed chunk_audio
            combined_audio = merge_audio(combined_audio, chunk_audio_without_silence)
      
        # Clear memory
        del chunk_audio
        gc.collect()
    
    # Normalize the final audio to prevent clipping
    # max_amplitude = np.max(np.abs(combined_audio))
    # if max_amplitude > 1.0:
    #     combined_audio /= max_amplitude
    
    # Write the final output file
    sf.write(output_file, combined_audio, TARGET_SAMPLE_RATE, format='wav')

if __name__ == "__main__":
    main()