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

if os.path.exists('/Users/tenchc/Documents/GitHub/takingstock/'):
    sys.path.insert(1, '/Users/tenchc/Documents/GitHub/takingstock/')
from mp_db_io import DataIO



TOPIC=11 # what folder are the files in?

# --- Batch mode config ---
BATCH_MODE = True          # set True to process every topic in BATCH_TOPICS
BATCH_TOPICS = [
"cc100_p2263_t0_1781354008.2844028",
"cc100_p733_t0_1781354008.284462",
"cc100_p734_t0_1781354008.284565",
"cc106_p2341_t0_1781354008.284698",
"cc129_p2263_t0_1781354008.284726",
"cc129_p2341_t0_1781354008.2849338",
"cc129_p586_t0_1781354040.627423",
"cc129_p727_t0_1781354043.751498",
"cc138_p1685_t0_1781354046.6547031",
"cc140_p2341_t0_1781354051.225558",
"cc140_p734_t0_1781354052.4769",
"cc153_p727_t0_1781354057.051112",
"cc164_p2263_t0_1781354079.1617",
"cc164_p586_t0_1781354080.3379",
"cc173_p2341_t0_1781354081.947423",
"cc173_p734_t0_1781354089.871104",
"cc174_p2341_t0_1781354091.804319",
"cc174_p734_t0_1781354092.7694058",
"cc176_p2341_t0_1781354112.926543",
"cc18_p1685_t0_1781354131.618468",
"cc18_p2263_t0_1781354132.692957",
"cc18_p3052_t0_1781354149.487799",
"cc18_p586_t0_1781354161.8375769",
"cc18_p727_t0_1781354163.805249",
"cc18_p733_t0_1781354166.225365",
"cc18_p734_t0_1781354168.5396721",
"cc181_p1685_t0_om1_1781354118.201664",
"cc181_p1685_t0_om2_1781354124.696685",
"cc181_p2230_t0_1781354129.427186",
"cc207_p2341_t0_1781354171.375859",
"cc207_p734_t0_1781354183.6664262",
"cc218_p1685_t0_1781354198.566848",
"cc221_p2341_t0_1781354205.85057",
"cc221_p258_t0_1781354208.625665",
"cc221_p727_t0_1781354209.569845",
"cc232_p1685_t0_om2_1781354211.2346082",
"cc240_p1685_t0_1781354217.0588748",
"cc252_p2263_t0_1781354230.850847",
"cc252_p2341_t0_1781354246.4691432",
"cc252_p586_t0_1781354247.595505",
"cc252_p727_t0_1781354249.5857542",
"cc252_p734_t0_1781354250.786443",
"cc272_p1685_t0_1781354251.9313269",
"cc272_p258_t0_1781354261.960465",
"cc276_p2263_t0_1781354281.597419",
"cc276_p258_t0_1781354288.6031659",
"cc276_p727_t0_1781354293.667905",
"cc276_p734_t0_1781354293.898411",
"cc285_p258_t0_1781354297.069623",
"cc290_p1685_t0_1781354300.060731",
"cc294_p727_t0_1781354317.528149",
"cc299_p2341_t0_om1_1781354331.248721",
"cc299_p2341_t0_om14-21_1781354322.510436",
"cc299_p2341_t0_om2_1781354331.619519",
"cc299_p2341_t0_om3-7+22_1781354334.404993",
"cc299_p2341_t0_om8-13_1781354337.1965609",
"cc299_p586_t0_1781354351.331156",
"cc299_p734_t0_1781354364.050156",
"cc32_p586_t0_1781354400.387258",
"cc322_p1685_t0_1781354365.240767",
"cc322_p727_t0_1781354369.3196921",
"cc329_p1685_t0_1781354375.070813",
"cc329_p727_t0_1781354377.228257",
"cc329_p733_t0_1781354387.1715121",
"cc336_p2263_t0_1781354406.687737",
"cc376_p734_t0_1781354408.5397918",
"cc396_p1685_t0_1781354411.103116",
"cc410_p2263_t0_1781354415.480688",
"cc410_p2341_t0_1781354420.493612",
"cc410_p734_t0_1781354436.586436",
"cc423_p1685_t0_1781354442.5726871",
"cc423_p258_t0_1781354445.3870761",
"cc460_p2263_t0_1781354450.339499",
"cc460_p727_t0_1781354452.262708",
"cc47_p2341_t0_1781354529.011594",
"cc47_p734_t0_1781354536.7478101",
"cc472_p2341_t0_om1_1781354472.9020731",
"cc472_p2341_t0_om14-21_1781354469.327456",
"cc472_p2341_t0_om2_1781354477.9506829",
"cc472_p734_t0_om1_1781354487.61303",
"cc472_p734_t0_om14-21_1781354482.0088792",
"cc472_p734_t0_om2_1781354488.780426",
"cc472_p734_t0_om8-13_1781354503.510774",
"cc474_p734_t0_1781354506.477726",
"cc479_p1685_t0_om1_1781354518.20244",
"cc479_p1685_t0_om2_1781354520.133919",
"cc479_p2230_t0_1781354521.776533",
"cc485_p2263_t0_1781354543.565963",
"cc487_p2263_t0_1781354551.327713",
"cc487_p2341_t0_1781354556.994349",
"cc487_p727_t0_1781354557.383531",
"cc490_p2263_t0_1781354574.891683",
"cc490_p2341_t0_1781354577.5796502",
"cc490_p727_t0_1781354579.8854342",
"cc490_p733_t0_1781354582.421941",
"cc518_p586_t0_1781354587.988336",
"cc523_p1685_t0_1781354609.543432",
"cc528_p1685_t0_1781354619.2121432",
"cc542_p2341_t0_1781354621.791092",
"cc547_p2341_t0_1781354621.9293509",
"cc547_p734_t0_1781354629.9248939",
"cc56_p1685_t0_om1_1781354665.714916",
"cc56_p1685_t0_om2_1781354674.618307",
"cc56_p258_t0_1781354678.4502988",
"cc566_p2341_t0_om1_1781354642.981844",
"cc566_p2341_t0_om14-21_1781354630.351407",
"cc566_p2341_t0_om2_1781354651.203597",
"cc566_p2341_t0_om3-7+22_1781354658.6624482",
"cc566_p2341_t0_om8-13_1781354659.610663",
"cc572_p734_t0_1781354695.865389",
"cc579_p2341_t0_1781354696.667917",
"cc587_p734_t0_1781354697.1006231",
"cc589_p2341_t0_1781354711.259119",
"cc638_p2341_t0_1781354713.486352",
"cc638_p734_t0_1781354717.318681",
"cc65_p2341_t0_1781354724.511369",
"cc701_p1685_t0_1781354730.3445451",
"cc701_p258_t0_1781354742.1572878",
"cc718_p2341_t0_om14-21_1781354750.796727",
"cc718_p2341_t0_om2_1781354759.047947",
"cc718_p2341_t0_om3-7+22_1781354762.185907",
"cc718_p734_t0_1781354771.3586729",
"cc720_p2341_t0_1781354781.573573",
"cc720_p727_t0_1781354795.340511",
"cc722_p2341_t0_1781354796.9534411",
"cc722_p734_t0_1781354797.845109",
"cc729_p734_t0_1781354810.9727528",
"cc749_p2341_t0_1781354817.0406358",
"cc749_p727_t0_1781354820.128577",
"cc749_p733_t0_1781354832.535018",
"cc752_p1685_t0_1781354835.697527",
"cc752_p2263_t0_1781354837.6508071",
"cc752_p2341_t0_1781354847.635193",
"cc752_p586_t0_1781354855.974784",
"cc752_p727_t0_om0_1781354864.7338738",
"cc752_p727_t0_om1_1781354868.186869",
"cc752_p727_t0_om14-21_1781354866.624934",
"cc752_p727_t0_om2_1781354877.295719",
"cc752_p727_t0_om8-13_1781354892.340652",
"cc752_p733_t0_1781354899.558799",
"cc757_p2341_t0_1781354901.525433",
"cc764_p2263_t0_1781354903.177483",
"cc83_p1685_t0_1781354907.8347661",
"cc84_p2341_t0_1781354915.344958",
"cc89_p258_t0_1781354926.761695",
"cc89_p727_t0_1781354939.339426"
]  # topics to process when BATCH_MODE = True
# -------------------------

CSV_FILE = f"metas_{TOPIC}.csv"  # overwritten per-topic when BATCH_MODE = True
SOUND_FOLDER = "tts_files_test"
SOUND_FOLDER = "tts_files_pitch_shift"
# SOUND_FOLDER = "37_metas_hold_for_now"

# TOPICFOLDER = "topic" + str(TOPIC)

# start = time.time()
######Michael's folders##########
io = DataIO()
INPUT = io.ROOTSSD # folder that holds SOUND_FOLDER and audiopduction folders
#################################

######Satyam's folders###########
# INPUT = "C:/Users/jhash/Documents/GitHub/facemap2/sound"
#################################

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
LOUD_ALOWED = 2
LOUD_RESET = 7
loud_counter = []
fake_loud = False
channel_counter = 0
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
        vol = scale_volume_linear(volume_fit, .02,.08)*cycler[0]
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
        # TEMP CHANGE (was: for key in KEYS[TOPIC]:)
        for key in KEYS[37]:
            if key in word:
                print(" ---- ", key, "found in", word, row['description'],row['image_id'])
                found_list.append(index)
                break
    if len(found_list)==0:
        # TEMP CHANGE (was: print("No keys found in", row['description'],"for topic model",KEYS[TOPIC]))
        print("No keys found in", row['description'],"for topic model",KEYS[11])
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

def build_quiet_background(quiet_files, total_duration, vol=0.04):
    """Build a full-duration looping quiet background layer (t=0 to total_duration).

    Cycles through *quiet_files* end-to-end, applying a gentle random pan to
    each clip, and returns a stereo numpy array ready to be mixed into the
    combined audio.  Using a fixed full-duration layer guarantees the murmur
    is always present regardless of where quiet-tier CSV rows fall.
    """
    if not quiet_files:
        return None

    files = list(dict.fromkeys(quiet_files))  # deduplicate, preserve order
    total_samples = int(total_duration * TARGET_SAMPLE_RATE)
    background = np.zeros((total_samples, 2))
    cursor = 0
    file_idx = 0
    consecutive_errors = 0

    while cursor < total_samples:
        if consecutive_errors >= len(files):
            print("build_quiet_background: all files unreadable, aborting pad")
            break
        filepath = files[file_idx % len(files)]
        file_idx += 1
        try:
            audio, sr = sf.read(filepath)
            consecutive_errors = 0
        except Exception as e:
            print(f"build_quiet_background: skipping {filepath}: {e}")
            consecutive_errors += 1
            continue

        audio, _ = conform_sample_rate(audio, sr)
        # apply a slight random volume wobble so the loop doesn't sound static
        clip_vol = vol * np.random.uniform(0.7, 1.3)
        audio = audio * clip_vol
        if len(audio.shape) == 1:
            audio = np.column_stack((audio, audio))
        # gentle pan: keep within centre range so the murmur doesn't pull hard
        pan = np.random.uniform(0.25, 0.75)
        audio[:, 0] *= (1.0 - pan)
        audio[:, 1] *= pan
        end = min(cursor + len(audio), total_samples)
        background[cursor:end] += audio[:end - cursor]
        cursor = end

    return background


def process_audio_chunk(chunk_df, existing_files, input_folder, start_index, chunk_index):
    left_channel_data = []
    right_channel_data = []
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

        if pd.notna(description) and str(image_id) in existing_files.keys():
            input_file = existing_files.get(str(image_id))
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

        input_path = os.path.join(INPUT,SOUND_FOLDER,input_file)


        # Read the audio file
        try:
            audio_data, sample_rate = sf.read(input_path)
        except Exception as e:
            print(f"Skipping corrupted/unreadable file {input_path}: {e}")
            continue
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
            # if len(loud_counter) is odd pan left, if even pan right
            # trying to alternate the loud audio from side to side
            if channel_counter % 2 == 1:
                pan = np.random.uniform(-1, 0)
            else: 
                pan = np.random.uniform(0, 1)
            channel_counter += 1

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

        # track quiet-tier (lowest volume) files so we can loop them later if needed
        if float(row['topic_fit']) < QUIET:
            quiet_files_used.append(input_path)
            quiet_max_end_time = max(quiet_max_end_time, end_time)
        
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
    
    # If no audio was collected (all rows skipped), return silence
    if not left_channel_data:
        print("process_audio_chunk: no audio collected for this chunk, returning silence")
        silence = np.zeros((TARGET_SAMPLE_RATE, 2))
        return silence, 0.0, quiet_files_used, quiet_max_end_time

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


def run_topic(topic):
    """Process a single topic and write its output file."""
    global TOPIC, CSV_FILE, OFFSET, existing_files, loud_counter, channel_counter, fake_loud

    # configure globals for this topic
    TOPIC = topic
    CSV_FILE = f"metas_{TOPIC}.csv"
    OFFSET = OFFSET_DICT.get(TOPIC, 0.0743)

    # reset stateful globals so each topic starts clean
    loud_counter = []
    channel_counter = 0
    fake_loud = False

    output_path = os.path.join(INPUT, f"multitrack_mixdown_offset_{TOPIC}.wav")
    if os.path.exists(output_path):
        print(f"[Topic {TOPIC}] Output already exists, skipping: {output_path}")
        return

    print(f"\n{'='*60}")
    print(f"[Topic {TOPIC}] Starting — CSV: {CSV_FILE}  OFFSET: {OFFSET}")
    chosen_keys = KEYS.get(TOPIC, None)
    if chosen_keys is not None:
        print(f"[Topic {TOPIC}] KEYS chosen: {chosen_keys}")
    else:
        print(f"[Topic {TOPIC}] WARNING: TOPIC {TOPIC} not found in KEYS dict — search_for_keys will fall back to KEYS[11]: {KEYS[11]}")
    print(f"{'='*60}")

    io = DataIO()
    df = pd.read_csv(os.path.join(INPUT, "audioproduction", CSV_FILE))

    raw_files = io.get_img_list(os.path.join(INPUT, SOUND_FOLDER))

    existing_files = {os.path.basename(f).split("_")[0]: f for f in raw_files}
    existing_files = {k: v for k, v in existing_files.items() if int(k) in df['image_id'].values}
    print(f"[Topic {TOPIC}] Existing files after INTERSECT:", len(existing_files))
    # TEMP CHANGE: print sample keys and paths to debug filename mismatch
    for k, v in list(existing_files.items())[:5]:
        print(f"  existing_files key: {repr(k)}  ->  {v}")

    combined_audio = None
    start_index = 0
    all_quiet_files = []      # accumulate quiet-tier file paths across all chunks
    quiet_coverage_end = 0.0  # track the furthest end time of any quiet-tier clip

    chunks = pd.read_csv(os.path.join(INPUT, "audioproduction", CSV_FILE), chunksize=CHUNK_SIZE)
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
            non_silent_index = int(np.floor(non_silent_index_raw / 2))
            print("Non-silent index:", non_silent_index)
            print("combined_audio shape:", combined_audio.shape, "chunk_audio shape:", chunk_audio.shape)
            np.set_printoptions(threshold=100)
            print(chunk_audio[:non_silent_index])
            print(chunk_audio[non_silent_index:])
            chunk_audio_without_silence = chunk_audio[non_silent_index:]
            combined_audio = merge_audio(combined_audio, chunk_audio_without_silence)
        del chunk_audio
        gc.collect()

    # --- Quiet-tier barrier: lay a continuous looping murmur under the entire track ---
    # Rather than detecting when quiet audio ends, we always build a full-duration
    # quiet background so the murmur is guaranteed to persist from start to finish.
    total_duration = len(combined_audio) / TARGET_SAMPLE_RATE
    print(f"[Topic {TOPIC}] Quiet tier reached {quiet_coverage_end:.1f}s / {total_duration:.1f}s total")
    if all_quiet_files:
        print(f"[Topic {TOPIC}] Building full-duration quiet background ({total_duration:.1f}s, "
              f"{len(set(all_quiet_files))} unique files)…")
        quiet_bg = build_quiet_background(all_quiet_files, total_duration)
        if quiet_bg is not None:
            if len(quiet_bg) > len(combined_audio):
                combined_audio = np.pad(
                    combined_audio,
                    ((0, len(quiet_bg) - len(combined_audio)), (0, 0)),
                    'constant',
                )
            combined_audio[:len(quiet_bg)] += quiet_bg
            print(f"[Topic {TOPIC}] Quiet background mixed in ({len(quiet_bg)/TARGET_SAMPLE_RATE:.1f}s)")
    else:
        print(f"[Topic {TOPIC}] No quiet-tier files collected — skipping quiet background")

    print(f"[Topic {TOPIC}] Combined audio shape before writing:", combined_audio.shape)
    print(f"[Topic {TOPIC}] Writing to file:", output_path)
    sf.write(output_path, combined_audio, TARGET_SAMPLE_RATE, format='wav')
    del combined_audio
    gc.collect()


def main():
    if BATCH_MODE:
        print(f"Batch mode ON — processing topics: {BATCH_TOPICS}")
        for topic in BATCH_TOPICS:
            run_topic(topic)
        print("\nBatch complete.")
    else:
        run_topic(TOPIC)

if __name__ == "__main__":
    main()