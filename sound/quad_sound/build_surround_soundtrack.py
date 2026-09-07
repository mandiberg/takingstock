"""Quadraphonic soundtrack mixer.

Same pipeline as build_soundtrack.py, but each clip is placed on 2, 3, or 4
speakers of a quad square according to the existing quiet / mid / loud volume
branches. Output is a 4-channel WAV (FL FR BL BR).
"""
import pandas as pd
import os
import csv
import shutil
import subprocess
import time
import soundfile as sf
import numpy as np
import librosa
import gc

# go get IO class from parent folder
# caution: path[0] is reserved for script path (or '' in REPL)
import sys
if sys.platform == "darwin": sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
elif sys.platform == "win32": sys.path.insert(1, 'C:/Users/jhash/Documents/GitHub/facemap2/')

if os.path.exists('/Users/tenchc/Documents/GitHub/takingstock/'):
    sys.path.insert(1, '/Users/tenchc/Documents/GitHub/takingstock/')
from mp_db_io import DataIO


######Michael's folders##########
io = DataIO()
INPUT = io.ROOTSSD # folder that holds SOUND_FOLDER and audiopduction folders
#################################

######Tench's folders###########
INPUT = "/Volumes/OWC5/tts_sport"
#################################

TOPIC = 0  # non-batch only: which metas_{TOPIC}.csv to mix
# KEYS lists to union in search_for_keys (both batch and non-batch).
KEY_TOPICS = [0, 3, 15, 45]

# --- Batch mode config ---
BATCH_MODE = True          # set True to process cluster folders under BATCH_FOLDER_NAME
# Parent folder (under INPUT, or absolute) whose subfolders each contain metas.csv.
# Example layout:
#   BATCH_FOLDER_NAME/clustercc1_p1_t0_om1_1788371815.2307808/metas.csv
BATCH_FOLDER_NAME = "/Volumes/OWC5/tts_sport/test_clusters"
# Optional subset: folder names under BATCH_FOLDER_NAME, or absolute cluster paths.
# Empty list = every subfolder that contains metas.csv.
BATCH_CLUSTERS = [
    # "clustercc1_p1_t0_om1_1788371815.2307808",
    # "clustercc8_p1_t0_om1_1788402205.695117",
]
METAS_CSV_NAME = "metas.csv"
# Cluster metas.csv has no audio filename; the last field is topic weight.
METAS_COLUMNS = ["image_id", "description", "topic_fit", "detections", "object", "topic"]
METAS_AUDIO_COLUMNS = [
    "image_id", "description", "topic_fit", "detections", "objects", "topic", "filename",
]
# -------------------------

CSV_FILE = f"metas_{TOPIC}.csv"  # overwritten per-topic when BATCH_MODE = True
SOUND_FOLDER = "."
# SOUND_FOLDER = "37_metas_hold_for_now"
METAS_AUDIO_CSV = os.path.join(INPUT, "metas_audio.csv")
MISSING_IDS_CSV = os.path.join(INPUT, "missing_ids.csv")

# TOPICFOLDER = "topic" + str(TOPIC)

# start = time.time()


# Choose a file starting with a given string
# prefixed = [filename for filename in os.listdir('.') if filename.startswith("prefix")]

# df and existing_files are loaded per-topic inside main()

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
OFFSET_DICT = {
    23: 0.075, # T23
    32: 0.0697, # T32
    37: 0.0755, # T37
}

OFFSET = OFFSET_DICT.get(TOPIC, 0.0743)  # overwritten per-topic when BATCH_MODE = True
#### GENERALLY 3 words per second
WPS=4
SCALE_EXPONENT = 0 # exponent for scale_volume_exp; 0 = linear, 1 = cubic
VOLUME_MIN = 0
VOLUME_MAX = .8
FIT_VOL_MIN = .1
FIT_VOL_MAX = 1
FADEOUT = 7
FADE_TIME = 1
QUIET =.5
# Quiet-tier volume range (matches scale_volume quiet branch).
QUIET_VOL_MIN = .02
QUIET_VOL_MAX = .08
QUIET_PAD_FADE_IN = 3.0  # seconds to crossfade the tail pad into existing quiet
QUIET_PAD_FADEOUT = 15   # per-clip fadeout, same default as scale_volume
LOUD_ALOWED = 2
LOUD_RESET = 7
loud_counter = []
fake_loud = False
channel_counter = 0

# WAV / FFmpeg quad channel order: FL, FR, BL, BR
N_CHANNELS = 4
SPEAKER_NAMES = ("FL", "FR", "BL", "BR")
# Clockwise around the room as indices into SPEAKERS / channels
CYCLE = (0, 1, 3, 2)  # FL, FR, BR, BL
# Fraction of total gain the anchor speaker gets (inclusive). Remainder is
# split randomly among the other speakers in that tier.
ANCHOR_RANGE_MID = [0.50, 0.75]
ANCHOR_RANGE_LOUD = [0.40, 0.70]
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
        path = existing_files.get(image_id_key(image_id))
        # if path containts meta, return True
        # TEMP CHANGE (was: if "bark_v5" in path: return True)
        if path is None: return False
        if "bark_v5" in path: return True
        else: return False

    global loud_counter
    global fake_loud
    volume_fit = float(row['topic_fit'])  # Using topic_fit as the volume level 
    # defaults
    fadein = 0
    fadeout = 15

    # search_for_keys to see where the matching keys are
    key_index,desc_count=search_for_keys(row)

    if volume_fit < QUIET:
        # vol = scale_volume_exp(volume_fit, 3)
        vol = scale_volume_linear(volume_fit, QUIET_VOL_MIN, QUIET_VOL_MAX)*cycler[0]
        # vol = .001
    elif len(key_index)>0:
        # if keys are found, set the volume and fade in out based on the keys found
        fadein,fadeout=calculate_fades(key_index,desc_count, audio_data, sample_rate)
        vol = scale_volume_exp(volume_fit,SCALE_EXPONENT)*1
        print(key_index)
        # start,end=key_index[0],key_index[-1]
        # vol =0
        # if vol < .5: vol = .001
        if vol < QUIET: 
            if vol > QUIET/2:
                # vol = vol - len(loud_counter)*.1
                if len(loud_counter) == 0:
                    if not fake_loud:
                        # trying to only trigger this once per loud_counter cycle
                        fake_loud = True
                        print("ffffffff    Fake loud set")

                        # if there are no loud files, scale the volume between .4 and .8
                        # to fill silence
                        vol = scale_volume_linear(volume_fit,0 ,.8)
                    else:
                        vol = (vol*.35) *cycler[1]
                        # vol = vol / (len(loud_counter)*.5+1)
                        # vol = .001
                else:
                    # reduce the volume of the audio based on the number of loud files
                    vol = vol / (len(loud_counter)*.5+1)
                if np.max(np.abs(audio_data)) > .8: vol = vol/3
                # if vol > .8: vol = .8
                # vol = .001
            else:
                vol = (vol*.45) *cycler[1]
                # vol = .001
        elif is_bark_loud(row):
            if np.max(np.abs(audio_data)) > QUIET: vol = vol/3
        # else: vol = .001
    else:
        vol = scale_volume_linear(volume_fit, .04,.15)*cycler[1]
        # if vol > .1: vol = .1
        # vol = vol*cycler[1]
        print("cylcerl vol",vol)
        # vol = .001
    return vol, fadeout,fadein


def to_mono(audio):
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def spatial_tier(row):
    """Match scale_volume branches: quiet / loud(keys) / mid."""
    volume_fit = float(row["topic_fit"])
    if volume_fit < QUIET:
        return "quiet"
    key_index, _ = search_for_keys(row)
    if len(key_index) > 0:
        return "loud"
    return "mid"


def random_normalized_weights(n):
    w = np.random.random(n)
    s = w.sum()
    if s <= 0:
        return np.ones(n) / n
    return w / s


def _anchor_share(range_lo_hi):
    lo, hi = range_lo_hi
    return float(np.random.uniform(lo, hi))


def spatial_gains(tier):
    """Weights for FL, FR, BL, BR that sum to 1 over the chosen speakers.

    Returns (gains, anchor_info). anchor_info is None for quiet, else
    (speaker_name, share, [lo, hi]).
    """
    gains = np.zeros(N_CHANNELS)
    start = np.random.randint(0, 4)
    if tier == "quiet":
        idxs = [CYCLE[start], CYCLE[(start + 1) % 4]]
        weights = random_normalized_weights(len(idxs))
        for idx, weight in zip(idxs, weights):
            gains[idx] = weight
        return gains, None

    if tier == "mid":
        anchor = CYCLE[start]
        others = [CYCLE[(start - 1) % 4], CYCLE[(start + 1) % 4]]
        lo_hi = ANCHOR_RANGE_MID
        share = _anchor_share(lo_hi)
        remainder = random_normalized_weights(2) * (1.0 - share)
        gains[anchor] = share
        for idx, weight in zip(others, remainder):
            gains[idx] = weight
        return gains, (SPEAKER_NAMES[anchor], share, lo_hi)

    # loud: any speaker as anchor; remainder split among the other three
    anchor = start
    others = [i for i in range(N_CHANNELS) if i != anchor]
    lo_hi = ANCHOR_RANGE_LOUD
    share = _anchor_share(lo_hi)
    remainder = random_normalized_weights(3) * (1.0 - share)
    gains[anchor] = share
    for idx, weight in zip(others, remainder):
        gains[idx] = weight
    return gains, (SPEAKER_NAMES[anchor], share, lo_hi)


def apply_quad_gains(mono, gains):
    return np.column_stack([mono * g for g in gains])


def tag_quad_wav(path):
    """Stamp WAV channel_layout=quad without rematrixing.

    ffmpeg's default guess for an untagged 4-channel file is 4.0 (FL FR FC BC).
    If we only set the *output* layout to quad, ffmpeg remixes ch2 (our BL)
    into the front as center, which makes FL/FR dominate. Declaring the input
    as quad already keeps sample order FL FR BL BR and only writes the tag.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found — wrote 4-channel WAV without a quad channel_layout tag")
        return
    tmp = path + ".quadtmp.wav"
    try:
        subprocess.run(
            [
                ffmpeg, "-y",
                "-channel_layout", "quad",
                "-i", path,
                "-c", "copy",
                tmp,
            ],
            check=True,
            capture_output=True,
        )
        os.replace(tmp, path)
        print(f"Tagged quad channel_layout: {path}")
    except Exception as e:
        print(f"ffmpeg quad tag skipped: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)

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

def keys_for_search():
    """Union of KEYS stems for every id in KEY_TOPICS, first occurrence kept."""
    stems = []
    seen = set()
    for t in KEY_TOPICS:
        for key in KEYS.get(t, []):
            if key not in seen:
                seen.add(key)
                stems.append(key)
    return stems


def search_for_keys(row):
    # search the first three words of the description for each key in KEYS
    # if any of the keys are found, set the volume to 1
    # if not, set the volume to 0.5
    if pd.isna(row['description']): return [],0

    # found = False
    found_list=[]
    desc_split=row['description'].lower().split(" ")
    desc_count=len(desc_split)
    active_keys = keys_for_search()
    for index,word in enumerate(desc_split):
        for key in active_keys:
            if key in word:
                print(" ---- ", key, "found in", word, row['description'],row['image_id'])
                found_list.append(index)
                break
    if len(found_list)==0:
        print("No keys found in", row['description'],"for topic models", KEY_TOPICS)
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

# existing_files is populated per-topic inside main()
existing_files = {}

AUDIO_EXTS = {".wav", ".mp3", ".flac"}
HASH_TOP = tuple("0123456789ABCDEF")


def sound_dir():
    return os.path.normpath(os.path.join(INPUT, SOUND_FOLDER))


def hashed_audio_path(root, filename):
    """Two-level MD5 folders keyed on the full filename, including extension."""
    filename = os.path.basename(filename)
    level1, level2 = io.get_hash_folders(filename)
    return os.path.join(root, level1, level2, filename)


def resolve_audio_path(filename):
    """Prefer hashed layout; fall back to a flat file under SOUND_FOLDER."""
    root = sound_dir()
    filename = os.path.basename(filename)
    hashed = hashed_audio_path(root, filename)
    if os.path.isfile(hashed):
        return hashed
    flat = os.path.join(root, filename)
    if os.path.isfile(flat):
        return flat
    return hashed


def image_id_key(value):
    try:
        if pd.isna(value):
            return None
        return str(int(float(value)))
    except (TypeError, ValueError):
        return None


def _clean_filename(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    name = os.path.basename(str(value).strip())
    if not name or name.lower() in ("nan", "none"):
        return None
    return name


def load_filenames_from_metas_audio(path):
    """image_id -> basename from metas_audio.csv (last non-empty filename wins)."""
    if not os.path.isfile(path):
        print(f"metas_audio.csv not found: {path}")
        return {}
    df = pd.read_csv(path)
    name_col = "filename" if "filename" in df.columns else "out_name" if "out_name" in df.columns else None
    if name_col is None:
        print(f"{path} has no 'filename' or 'out_name' column")
        return {}
    df["image_id"] = pd.to_numeric(df["image_id"], errors="coerce")
    df[name_col] = df[name_col].map(_clean_filename)
    df = df.dropna(subset=["image_id", name_col])
    df["image_id"] = df["image_id"].astype(int)
    df = df.drop_duplicates(subset="image_id", keep="last")
    by_id = {str(iid): fname for iid, fname in zip(df["image_id"], df[name_col])}
    print(f"Loaded {len(by_id)} filenames from {path}")
    return by_id


def existing_audio_by_id(folder):
    """Walk hash trees (or the whole folder) and map image_id -> basename."""
    lists = walk_audio_filenames_by_id(folder)
    return {iid: pick_audio_filename(names) for iid, names in lists.items()}


def hash_dir_roots(folder):
    """Top-level 0-9/A-F hash directories under folder."""
    roots = []
    for top in HASH_TOP:
        path = os.path.join(folder, top)
        if os.path.isdir(path):
            roots.append(path)
    return roots


def walk_audio_filenames_by_id(folder):
    """image_id -> [basenames] from hash folders, or a full walk if none exist."""
    by_id = {}
    if not os.path.isdir(folder):
        return by_id
    roots = hash_dir_roots(folder)
    if not roots:
        roots = [folder]
    n_files = 0
    for root in roots:
        for walk_root, _dirs, files in os.walk(root):
            if "_x" in walk_root.split(os.sep):
                continue
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in AUDIO_EXTS:
                    continue
                key = image_id_key(os.path.splitext(fname)[0].split("_")[0])
                if key is None:
                    continue
                n_files += 1
                by_id.setdefault(key, [])
                if fname not in by_id[key]:
                    by_id[key].append(fname)
    print(f"  scrape indexed {n_files} audio file(s) for {len(by_id)} image_id(s)")
    return by_id


def pick_audio_filename(names):
    """Prefer wav, then coqui, then a stable last name."""
    names = [n for n in names if n]
    if not names:
        return None

    def score(name):
        lower = name.lower()
        return (
            1 if lower.endswith(".wav") else 0,
            1 if "_coqui_" in lower else 0,
            name,
        )

    return sorted(names, key=score)[-1]


def audio_on_disk(filename):
    filename = _clean_filename(filename)
    if not filename:
        return False
    path = resolve_audio_path(filename)
    return os.path.isfile(path)


_filenames_from_csv = None
_walked_audio_by_id = None
_walked_audio_all = None
_metas_audio_fieldnames = None


def filenames_from_metas_audio():
    global _filenames_from_csv
    if _filenames_from_csv is None:
        _filenames_from_csv = load_filenames_from_metas_audio(METAS_AUDIO_CSV)
    return _filenames_from_csv


def walked_audio_by_id():
    """Walk hash folders once per process; keep every clip per image_id."""
    global _walked_audio_by_id, _walked_audio_all
    if _walked_audio_by_id is None:
        root = sound_dir()
        print(f"Scraping hash-folder audio in {root} …")
        _walked_audio_all = walk_audio_filenames_by_id(root)
        _walked_audio_by_id = {
            iid: pick_audio_filename(names) for iid, names in _walked_audio_all.items()
        }
        print(f"Walk found {len(_walked_audio_by_id)} image_id(s) with audio")
    return _walked_audio_by_id


def metas_audio_fieldnames():
    """Header of metas_audio.csv, guaranteeing a filename column."""
    global _metas_audio_fieldnames
    if _metas_audio_fieldnames is not None:
        return _metas_audio_fieldnames
    names = list(METAS_AUDIO_COLUMNS)
    if os.path.isfile(METAS_AUDIO_CSV) and os.path.getsize(METAS_AUDIO_CSV) > 0:
        with open(METAS_AUDIO_CSV, "r", encoding="utf-8-sig", newline="") as f:
            header = list(csv.DictReader(f).fieldnames or [])
        if header:
            names = header
    if "filename" not in names:
        names.append("filename")
    _metas_audio_fieldnames = names
    return names


def _csv_cell(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return value


def cluster_row_to_metas_audio(row, filename, fieldnames):
    """Map a cluster metas.csv row onto metas_audio.csv columns + scraped filename."""
    data = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    objects = data.get("objects", data.get("object", ""))
    weight = data.get("weight", "")
    if weight is None or (isinstance(weight, float) and pd.isna(weight)):
        maybe = data.get("filename")
        if maybe is not None and _clean_filename(maybe) is None:
            weight = maybe
    out = {}
    for key in fieldnames:
        if key == "filename":
            out[key] = filename
        elif key == "objects":
            out[key] = _csv_cell(objects)
        elif key == "weight":
            out[key] = _csv_cell(weight)
        elif key == "object":
            out[key] = _csv_cell(data.get("object", objects))
        else:
            out[key] = _csv_cell(data.get(key, ""))
    return out


def _ensure_csv_trailing_newline(path):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) != b"\n":
            f.write(b"\n")


def append_metas_audio_rows(rows):
    """Append scraped cluster rows (with filename) to metas_audio.csv."""
    if not rows:
        return 0
    fieldnames = metas_audio_fieldnames()
    path = METAS_AUDIO_CSV
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    if exists:
        _ensure_csv_trailing_newline(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def append_scraped_cluster_rows(df, scraped):
    """Write metas.csv data + scraped filename for ids not yet in metas_audio.csv."""
    global _filenames_from_csv
    if not scraped:
        return 0
    csv_names = filenames_from_metas_audio()
    keys = df["image_id"].map(image_id_key)
    fieldnames = metas_audio_fieldnames()
    rows = []
    for iid, filename in scraped:
        if iid in csv_names:
            continue
        matches = df[keys == iid]
        if matches.empty:
            continue
        row = cluster_row_to_metas_audio(matches.iloc[-1], filename, fieldnames)
        rows.append(row)
        csv_names[iid] = filename
    written = append_metas_audio_rows(rows)
    if _filenames_from_csv is not None:
        _filenames_from_csv.update(csv_names)
    return written


_missing_ids_written = 0
_missing_ids_fieldnames = None


def index_audio_for_topic(df):
    """Resolve audio for this cluster's ids; scrape hash folders; backfill metas_audio.csv.

    Returns (existing, still_missing). existing maps image_id -> audio basename.
    Ids found on disk but missing from metas_audio.csv are appended with the
    cluster metas.csv fields plus the scraped filename.
    """
    csv_names = filenames_from_metas_audio()
    walked = walked_audio_by_id()
    df_ids = {k for k in (image_id_key(v) for v in df["image_id"]) if k is not None}
    existing = {}
    still_missing = []
    scraped = []
    csv_ok = 0
    csv_stale = 0
    for iid in df_ids:
        csv_name = _clean_filename(csv_names.get(iid))
        if csv_name and audio_on_disk(csv_name):
            existing[iid] = csv_name
            csv_ok += 1
            continue
        if csv_name:
            csv_stale += 1
        walked_name = _clean_filename(walked.get(iid))
        if walked_name:
            existing[iid] = walked_name
            if iid not in csv_names:
                scraped.append((iid, walked_name))
            continue
        still_missing.append(iid)

    appended = append_scraped_cluster_rows(df, scraped)
    print(f"  metas_audio.csv hits on disk: {csv_ok}; "
          f"csv filename missing on disk: {csv_stale}; "
          f"scrape filled: {len(scraped)}; still missing: {len(still_missing)}")
    if appended:
        print(f"  appended {appended} row(s) to {METAS_AUDIO_CSV}")
    return existing, still_missing


def collect_missing_id_rows(df, missing_ids, cluster):
    """Append metas rows for ids with no audio to missing_ids.csv immediately."""
    global _missing_ids_written, _missing_ids_fieldnames
    if not missing_ids:
        return 0
    missing_set = set(missing_ids)
    keys = df["image_id"].map(image_id_key)
    rows = df[keys.isin(missing_set)].copy()
    if rows.empty:
        return 0
    rows.insert(0, "cluster", cluster)
    path = MISSING_IDS_CSV
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    if exists:
        _ensure_csv_trailing_newline(path)
        if _missing_ids_fieldnames is None:
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                _missing_ids_fieldnames = list(csv.DictReader(f).fieldnames or [])
        fieldnames = _missing_ids_fieldnames
        for col in fieldnames:
            if col not in rows.columns:
                rows[col] = ""
        rows = rows[fieldnames]
    else:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        _missing_ids_fieldnames = list(rows.columns)
    rows.to_csv(path, mode="a", header=not exists, index=False)
    n = len(rows)
    _missing_ids_written += n
    print(f"  appended {n} missing-id row(s) to {path}")
    return n


def write_missing_ids_csv(path=None):
    """Summarize missing_ids.csv appends for this run."""
    path = path or MISSING_IDS_CSV
    if _missing_ids_written == 0:
        print(f"No missing ids appended to {path}")
        return
    print(f"Appended {_missing_ids_written} missing-id row(s) this run to {path}")


def _load_quiet_clips(quiet_files):
    """Load unique quiet files to mono at TARGET_SAMPLE_RATE."""
    clips = []
    for filepath in dict.fromkeys(quiet_files):
        try:
            audio, sr = sf.read(filepath)
        except Exception as e:
            print(f"build_quiet_background: skipping {filepath}: {e}")
            continue
        audio, _ = conform_sample_rate(to_mono(audio), sr)
        if len(audio) == 0:
            continue
        clips.append(audio)
    return clips


def _fadeout_mono(audio, duration, sample_rate=TARGET_SAMPLE_RATE):
    """Squared fadeout without the verbose apply_fadeout prints."""
    duration = check_fade_length(duration, audio, sample_rate)
    length = int(duration * sample_rate)
    if length <= 0:
        return audio
    end = audio.shape[0]
    start = end - length
    fade_curve = np.power(np.linspace(1.0, 0.0, length), 2)
    audio[start:end] *= fade_curve
    return audio


def build_quiet_background(quiet_files, total_duration, start_time=0.0, offset=None,
                           fade_in=QUIET_PAD_FADE_IN):
    """Overlapping quiet bed from start_time to total_duration on the OFFSET grid.

    Matches the main mixer: a new clip every OFFSET seconds, quiet-tier volume
    and two-adjacent-speaker placement. Starts *fade_in* seconds before
    start_time so the pad crossfades as the original quiet layer dies out.
    The file pool is shuffled on every pass so the tail is not a literal repeat.
    """
    if not quiet_files or total_duration <= 0:
        return None

    offset = OFFSET if offset is None else offset
    clips = _load_quiet_clips(quiet_files)
    if not clips:
        print("build_quiet_background: no readable quiet files, aborting pad")
        return None

    pad_start = max(0.0, start_time - fade_in)
    if pad_start >= total_duration:
        return None

    total_samples = int(total_duration * TARGET_SAMPLE_RATE)
    background = np.zeros((total_samples, N_CHANNELS))

    n = len(clips)
    order = np.arange(n)
    np.random.shuffle(order)
    order_pos = 0

    t = pad_start
    n_placed = 0
    while t < total_duration:
        clip = clips[order[order_pos]]
        order_pos += 1
        if order_pos >= n:
            np.random.shuffle(order)
            order_pos = 0

        clip_vol = np.random.uniform(QUIET_VOL_MIN, QUIET_VOL_MAX)
        mono = _fadeout_mono(clip * clip_vol, QUIET_PAD_FADEOUT)
        gains, _anchor = spatial_gains("quiet")
        audio = apply_quad_gains(mono, gains)

        start_sample = int(t * TARGET_SAMPLE_RATE)
        if start_sample >= total_samples:
            break
        end_sample = min(start_sample + len(audio), total_samples)
        n_copy = end_sample - start_sample
        if n_copy > 0:
            background[start_sample:end_sample] += audio[:n_copy]
            n_placed += 1
        t += offset

    fade_begin = int(pad_start * TARGET_SAMPLE_RATE)
    fade_samples = int(fade_in * TARGET_SAMPLE_RATE)
    fade_end = min(fade_begin + fade_samples, total_samples)
    n_fade = fade_end - fade_begin
    if n_fade > 1:
        fade_curve = np.power(np.linspace(0.0, 1.0, n_fade), 2)
        background[fade_begin:fade_end] *= fade_curve[:, np.newaxis]

    print(f"build_quiet_background: placed {n_placed} overlapping clips "
          f"from {pad_start:.1f}s to {total_duration:.1f}s "
          f"(offset={offset:.4f}s, {n} unique files)")
    return background


def process_audio_chunk(chunk_df, existing_files, input_folder, start_index, chunk_index):
    channel_data = [[] for _ in range(N_CHANNELS)]
    quiet_files_used = []   # paths of files placed in the quiet tier
    quiet_max_end_time = 0  # latest end time seen for a quiet-tier clip
    max_end_time = 0
    global loud_counter
    global channel_counter
    global fake_loud
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

        iid = image_id_key(image_id)
        if pd.notna(description) and iid is not None and iid in existing_files:
            input_file = existing_files.get(iid)
            print("Using existing file:", input_file)
        elif pd.notna(description) and image_id:
            if not existing_files:
                print(f"Skipping image_id {image_id}: no existing files to fall back to")
                continue
            input_file = np.random.choice(list(existing_files.values()))
            if row['topic_fit'] < .6:
                print("unprocessed meta file")
            elif row['topic_fit'] > .75:
                print("unprocessed openai file")
        elif pd.isna(description) and image_id:
            if not existing_files:
                print(f"Skipping image_id {image_id} (NaN description): no existing files to fall back to")
                continue
            input_file = np.random.choice(list(existing_files.values()))
            if row['topic_fit'] > QUIET:
                row['topic_fit'] = row['topic_fit']/2
            print(f"is NaN assigned random file: {input_file} and topic_fit halved to {row['topic_fit']}")
        else:
            print("No good files found")
            continue

        input_path = resolve_audio_path(input_file)


        # Read the audio file
        try:
            audio_data, sample_rate = sf.read(input_path)
        except Exception as e:
            print(f"Skipping corrupted/unreadable file {input_path}: {e}")
            continue
        print("length at start",len(audio_data))
        print("location",input_path)
        audio_data = to_mono(audio_data)
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
            if fake_loud and len(loud_counter) == 0:
                # if loud_counter cycle is complete, reset fake_loud
                fake_loud = False
                print("rrrrrrrrrrrrr    Fake loud reset")
            if len(loud_counter) > LOUD_ALOWED*LOUD_RESET:
                # reset the loud counter if it gets too long
                loud_counter = []
        if loud_counter and len(loud_counter) >= LOUD_ALOWED and volume_scale > QUIET:
            # audio_data_adjusted = audio_data_adjusted* (1/len(loud_counter))


            if len(loud_counter) > LOUD_ALOWED*2:
                    # if there is a backlog of loud files, reduce the volume and play normal speed
                    # otherwise, the track will be 3x long, with the last 2x all the loud files
                # loud_divisor = min(len(loud_counter), 10)
                audio_data_adjusted = audio_data_adjusted * (1 / (2 + len(loud_counter)))
            else:
                print("TOO LOUD")
                loud_delay_duration = 2* (len(loud_counter)-LOUD_ALOWED)
                loud_offset = (max(loud_counter)/OFFSET) +(loud_delay_duration/OFFSET)
                print("Loud offset:", loud_offset)
        if volume_scale > QUIET:
            loud_counter.append(len(audio_data)/sample_rate)
            channel_counter += 1

        # Apply fadeout to the audio data
        apply_fadeout(audio_data_adjusted, sample_rate, fadeout)
        ################
        # Apply fadein to the audio data
        if fadein>0:apply_fadein(audio_data_adjusted, sample_rate, fadein)
        ####################
        tier = spatial_tier(row)
        gains, anchor_info = spatial_gains(tier)
        audio_data_adjusted = apply_quad_gains(audio_data_adjusted, gains)
        if anchor_info is not None:
            name, share, lo_hi = anchor_info
            print(f"Anchor {name}={share:.2f} range=[{lo_hi[0]:.2f}, {lo_hi[1]:.2f}] ({tier})")
        print(f"Spatial {tier}:", " ".join(
            f"{name}={g:.2f}" for name, g in zip(SPEAKER_NAMES, gains) if g > 1e-6
        ))

        # # Append audio data to respective lists
        # left_channel_data.append(audio_data_adjusted[:, 0])
        # right_channel_data.append(audio_data_adjusted[:, 1])
        repeat, last_description = test_repeat(description, last_description)
        # Calculate the start time for this audio clip
        # if repeat, then start at the same time as the last clip
        start_time = (start_index + i - repeat + loud_offset) * OFFSET
        end_time = start_time + len(audio_data_adjusted) / TARGET_SAMPLE_RATE
        max_end_time = max(max_end_time, end_time)

        # track quiet-tier (lowest volume) files so we can loop them later if needed
        if float(row['topic_fit']) < QUIET:
            quiet_files_used.append(input_path)
            quiet_max_end_time = max(quiet_max_end_time, end_time)
        
        # Create arrays with the correct offset
        n_samples = int(np.ceil(end_time * TARGET_SAMPLE_RATE))
        placed = [np.zeros(n_samples) for _ in range(N_CHANNELS)]
        
        # Insert the audio data at the correct position
        start_sample = int(start_time * TARGET_SAMPLE_RATE)
        end_sample = min(start_sample + len(audio_data_adjusted), n_samples)
        n_copy = end_sample - start_sample
        for ch in range(N_CHANNELS):
            placed[ch][start_sample:end_sample] = audio_data_adjusted[:n_copy, ch]
            channel_data[ch].append(placed[ch])
    
    # If no audio was collected (all rows skipped), return silence
    if not channel_data[0]:
        print("process_audio_chunk: no audio collected for this chunk, returning silence")
        silence = np.zeros((TARGET_SAMPLE_RATE, N_CHANNELS))
        return silence, 0.0, quiet_files_used, quiet_max_end_time

    # Mix the audio data for the chunk
    max_length = max(len(data) for ch in channel_data for data in ch)
    mixed_audio = np.zeros((max_length, N_CHANNELS))
    
    n_clips = len(channel_data[0])
    for i in range(n_clips):
        for ch in range(N_CHANNELS):
            clip = channel_data[ch][i]
            mixed_audio[:len(clip), ch] += clip
    
    # Clear memory
    del channel_data
    gc.collect()
    
    # save the mixed audio to a file
    # output_file = os.path.join(INPUT, f"multitrack_mixdown_offset_{TOPIC}_{chunk_index}.wav")
    # sf.write(output_file, mixed_audio, TARGET_SAMPLE_RATE, format='wav')

    return mixed_audio, max_end_time, quiet_files_used, quiet_max_end_time

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
    def _pad_to(arr, n):
        if len(arr) >= n:
            return arr[:n]
        extra = n - len(arr)
        if arr.ndim == 1:
            return np.pad(arr, (0, extra), "constant")
        return np.pad(arr, ((0, extra), (0, 0)), "constant")

    combined_audio_last_10s = _pad_to(combined_audio_last_10s, overlap_samples)
    chunk_audio_first_10s = _pad_to(chunk_audio_first_10s, overlap_samples)

    # Mix the audio by adding the arrays together
    overlapped_segment = combined_audio_last_10s + chunk_audio_first_10s

    # Concatenate the mixed segment with the remaining parts of combined_audio and chunk_audio_without_silence
    combined_audio = np.concatenate((combined_audio[:-overlap_samples], overlapped_segment, chunk_audio_without_silence[overlap_samples:]))
    # sf.write(str(len(c ombined_audio))+"combined_audio.wav", combined_audio, TARGET_SAMPLE_RATE, format='wav')
    return combined_audio


def resolve_batch_folder(folder_name):
    """Return an absolute path to the parent folder of cluster directories."""
    if os.path.isabs(folder_name):
        return folder_name
    return os.path.join(INPUT, folder_name)


def metas_csv_has_header(path):
    """True if the first field of the first line is image_id."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        first = f.readline()
    if not first:
        return False
    first_field = first.split(",", 1)[0].strip().strip('"').lower()
    return first_field == "image_id"


def metas_read_csv_kwargs(path):
    """Kwargs so headerless cluster metas.csv files still expose named columns."""
    if metas_csv_has_header(path):
        return {}
    return {"header": None, "names": METAS_COLUMNS}


def normalize_metas_df(df):
    """Cluster metas.csv stores topic weight, not an audio filename."""
    if df is None or df.empty:
        return df
    if "weight" not in df.columns and "filename" in df.columns:
        audio_names = df["filename"].map(_clean_filename)
        if audio_names.notna().sum() == 0:
            df = df.rename(columns={"filename": "weight"})
    return df


def read_metas_csv(path, **kwargs):
    merged = metas_read_csv_kwargs(path)
    merged.update(kwargs)
    chunksize = merged.pop("chunksize", None)
    if chunksize:
        return (
            normalize_metas_df(chunk)
            for chunk in pd.read_csv(path, chunksize=chunksize, **merged)
        )
    return normalize_metas_df(pd.read_csv(path, **merged))


def cluster_csv_path(folder):
    return os.path.join(folder, METAS_CSV_NAME)


def resolve_cluster_folder(entry, parent):
    """Resolve a BATCH_CLUSTERS entry to an absolute cluster folder path."""
    if os.path.isabs(entry):
        return entry
    return os.path.join(parent, entry)


def list_cluster_jobs(parent, names=None):
    """Return (folder_name, metas.csv path) for each cluster folder to mix.

    If names is empty, every subdirectory of parent that contains metas.csv
    is included. Otherwise each name is a folder under parent, or an absolute
    cluster path.
    """
    if not os.path.isdir(parent):
        raise FileNotFoundError(f"Batch folder not found: {parent}")

    if names:
        candidates = [resolve_cluster_folder(n, parent) for n in names]
    else:
        candidates = [
            os.path.join(parent, name)
            for name in sorted(os.listdir(parent))
            if os.path.isdir(os.path.join(parent, name))
        ]

    jobs = []
    for folder in candidates:
        csv_path = cluster_csv_path(folder)
        label = os.path.basename(os.path.normpath(folder))
        if not os.path.isdir(folder):
            print(f"Skipping {label}: not a directory ({folder})")
            continue
        if not os.path.isfile(csv_path):
            print(f"Skipping {label}: no {METAS_CSV_NAME} in {folder}")
            continue
        jobs.append((label, csv_path))
    return jobs


def run_topic(topic, csv_path=None):
    """Process a single topic/cluster and write its output file."""
    global TOPIC, CSV_FILE, OFFSET, existing_files, loud_counter, channel_counter, fake_loud

    # configure globals for this topic
    TOPIC = topic
    if csv_path is None:
        CSV_FILE = f"metas_{TOPIC}.csv"
        csv_path = os.path.join(INPUT, "audioproduction", CSV_FILE)
    else:
        CSV_FILE = os.path.basename(csv_path)
    OFFSET = OFFSET_DICT.get(TOPIC, 0.0743)

    # reset stateful globals so each topic starts clean
    loud_counter = []
    channel_counter = 0
    fake_loud = False

    output_path = os.path.join(INPUT, f"multitrack_mixdown_offset_{TOPIC}_quad.wav")

    topic_t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[Topic {TOPIC}] Starting — CSV: {csv_path}  OFFSET: {OFFSET}")
    missing_key_topics = [t for t in KEY_TOPICS if t not in KEYS]
    if missing_key_topics:
        print(f"[Topic {TOPIC}] WARNING: KEY_TOPICS not in KEYS dict: {missing_key_topics}")
    print(f"[Topic {TOPIC}] KEY_TOPICS {KEY_TOPICS} → {keys_for_search()}")
    print(f"{'='*60}")

    df = read_metas_csv(csv_path)

    print(f"[Topic {TOPIC}] Resolving audio via metas_audio.csv + hash-folder scrape")
    existing_files, missing_ids = index_audio_for_topic(df)
    collect_missing_id_rows(df, missing_ids, TOPIC)
    print(f"[Topic {TOPIC}] Existing files after INTERSECT:", len(existing_files))
    for k, v in list(existing_files.items())[:5]:
        print(f"  existing_files key: {repr(k)}  ->  {resolve_audio_path(v)}")

    if os.path.exists(output_path):
        print(f"[Topic {TOPIC}] Output already exists, skipping: {output_path}")
        return None

    combined_audio = None
    start_index = 0
    all_quiet_files = []      # accumulate quiet-tier file paths across all chunks
    quiet_coverage_end = 0.0  # track the furthest end time of any quiet-tier clip

    chunks = read_metas_csv(csv_path, chunksize=CHUNK_SIZE)
    for chunk_index, chunk in enumerate(chunks):
        chunk_audio, chunk_end_time, chunk_quiet_files, chunk_quiet_end = process_audio_chunk(chunk, existing_files, INPUT, start_index, chunk_index)
        print(f"[Topic {TOPIC}] Chunk audio length/sample:", len(chunk_audio)/TARGET_SAMPLE_RATE, "Chunk end time:", chunk_end_time)

        # collect quiet-tier bookkeeping
        all_quiet_files.extend(chunk_quiet_files)
        quiet_coverage_end = max(quiet_coverage_end, chunk_quiet_end)

        if combined_audio is None:
            combined_audio = chunk_audio
            print(chunk_index, "Combined audio shape:", combined_audio.shape, "Chunk audio shape:", chunk_audio.shape)
        else:
            non_silent_index_raw = np.argmax(np.abs(chunk_audio) > 0)
            nch = chunk_audio.shape[1] if chunk_audio.ndim > 1 else 1
            non_silent_index = int(np.floor(non_silent_index_raw / nch))
            print("Non-silent index:", non_silent_index)
            print("combined_audio shape:", combined_audio.shape, "chunk_audio shape:", chunk_audio.shape)
            np.set_printoptions(threshold=100)
            print(chunk_audio[:non_silent_index])
            print(chunk_audio[non_silent_index:])
            chunk_audio_without_silence = chunk_audio[non_silent_index:]
            combined_audio = merge_audio(combined_audio, chunk_audio_without_silence)
        del chunk_audio
        gc.collect()

    # --- Quiet-tier tail pad: overlapping murmur from where original quiet dies out ---
    total_duration = len(combined_audio) / TARGET_SAMPLE_RATE
    print(f"[Topic {TOPIC}] Quiet tier reached {quiet_coverage_end:.1f}s / {total_duration:.1f}s total")
    gap = total_duration - quiet_coverage_end
    if all_quiet_files and gap > OFFSET:
        print(f"[Topic {TOPIC}] Building overlapping quiet pad ({gap:.1f}s gap, "
              f"{len(set(all_quiet_files))} unique files, offset={OFFSET}s)…")
        quiet_bg = build_quiet_background(
            all_quiet_files,
            total_duration,
            start_time=quiet_coverage_end,
        )
        if quiet_bg is not None:
            if len(quiet_bg) > len(combined_audio):
                combined_audio = np.pad(
                    combined_audio,
                    ((0, len(quiet_bg) - len(combined_audio)), (0, 0)),
                    'constant',
                )
            combined_audio[:len(quiet_bg)] += quiet_bg
            print(f"[Topic {TOPIC}] Quiet pad mixed in ({len(quiet_bg)/TARGET_SAMPLE_RATE:.1f}s)")
    elif not all_quiet_files:
        print(f"[Topic {TOPIC}] No quiet-tier files collected — skipping quiet pad")
    else:
        print(f"[Topic {TOPIC}] Quiet tier already covers the track, skipping pad")

    print(f"[Topic {TOPIC}] Combined audio shape before writing:", combined_audio.shape)
    print(f"[Topic {TOPIC}] Writing to file:", output_path)
    sf.write(output_path, combined_audio, TARGET_SAMPLE_RATE, format='wav')
    tag_quad_wav(output_path)
    elapsed = time.time() - topic_t0
    print(f"[Topic {TOPIC}] Time to process output file: {elapsed:.1f}s")
    del combined_audio
    gc.collect()
    return elapsed


def main():
    global _missing_ids_written, _missing_ids_fieldnames
    _missing_ids_written = 0
    _missing_ids_fieldnames = None
    filenames_from_metas_audio()
    walked_audio_by_id()
    elapsed_times = []
    if BATCH_MODE:
        batch_dir = resolve_batch_folder(BATCH_FOLDER_NAME)
        jobs = list_cluster_jobs(batch_dir, BATCH_CLUSTERS)
        print(f"Batch mode ON — found {len(jobs)} cluster folder(s) in {batch_dir}")
        if not jobs:
            print(f"No cluster folders with {METAS_CSV_NAME} found.")
            return
        for topic, csv_path in jobs:
            elapsed = run_topic(topic, csv_path=csv_path)
            if elapsed is not None:
                elapsed_times.append(elapsed)
        print("\nBatch complete.")
    else:
        elapsed = run_topic(TOPIC)
        if elapsed is not None:
            elapsed_times.append(elapsed)
    write_missing_ids_csv()
    if elapsed_times:
        avg = sum(elapsed_times) / len(elapsed_times)
        print(f"Average processing time per cluster: {avg:.1f}s "
              f"({len(elapsed_times)} cluster(s))")
    else:
        print("Average processing time per cluster: n/a (no clusters processed)")

if __name__ == "__main__":
    main()