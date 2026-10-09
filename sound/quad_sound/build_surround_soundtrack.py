"""Surround soundtrack mixer.

At startup, pick quad or stereo.
Quad places each clip on 2, 3, or 4 speakers (FL FR BL BR).
Stereo places the same quiet / mid / loud tiers in L/R, so a two-speaker
quad pair is not folded into a hard pan.
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
from pick import pick

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
INPUT = "/Volumes/OWC5/tts_office"
#################################

TOPIC = 0  # non-batch only: which metas_{TOPIC}.csv to mix
# KEYS lists to union in search_for_keys (both batch and non-batch).
KEY_TOPICS = [0, 3, 15, 45]

# --- Batch mode config ---
BATCH_MODE = True          # set True to process cluster folders under BATCH_FOLDER_NAME
# Parent folder (under INPUT, or absolute) whose subfolders each contain metas.csv.
# Example layout:
#   BATCH_FOLDER_NAME/clustercc1_p1_t0_om1_1788371815.2307808/metas.csv
BATCH_FOLDER_NAME = "/Volumes/OWC5/tts_office/office_thoma_metas_sept28/_Body_thoma_sept28"
# Optional subset: folder names under BATCH_FOLDER_NAME, or absolute cluster paths.
# Empty list = every subfolder that contains metas.csv.
BATCH_CLUSTERS = [
    # "clustercc1_p1_t0_om1_1788371815.2307808",
    # "clustercc8_p1_t0_om1_1788402205.695117",
]

SOUND_FOLDER = "."
# SOUND_FOLDER = "37_metas_hold_for_now"

# Sampling rate for the mixdown
sample_rate = None

class AudioIndex:
    """Cached audio filename lookup for one input folder."""
    AUDIO_EXTS = {".wav", ".mp3", ".flac"}
    HASH_TOP = tuple("0123456789ABCDEF")
    METAS_CSV_NAME = "metas.csv"
    METAS_COLUMNS = ["image_id", "description", "topic_fit", "detections", "object", "topic"]
    METAS_AUDIO_COLUMNS = ["image_id", "description", "topic_fit", "detections", "objects", "topic", "filename"]

    def __init__(self, input_dir, sound_folder=".", io=None):
        self.input_dir = input_dir
        self.sound_folder = sound_folder
        self.io = io
        self.metas_audio_csv = os.path.join(input_dir, "metas_audio.csv")
        self.missing_ids_csv = os.path.join(input_dir, "missing_ids.csv")
        self._filenames_from_csv = None
        self._walked_audio_by_id = None
        self._walked_audio_all = None
        self._metas_audio_fieldnames = None
        self.missing_ids_written = 0
        self.missing_ids_fieldnames = None

    def sound_dir(self):
        return os.path.normpath(os.path.join(self.input_dir, self.sound_folder))

    def hashed_audio_path(self, root, filename):
        """Two-level MD5 folders keyed on the full filename, including extension."""
        filename = os.path.basename(filename)
        level1, level2 = self.io.get_hash_folders(filename)
        return os.path.join(root, level1, level2, filename)

    def resolve_audio_path(self, filename):
        """Prefer hashed layout; fall back to a flat file under self.sound_folder."""
        root = self.sound_dir()
        filename = os.path.basename(filename)
        hashed = self.hashed_audio_path(root, filename)
        if os.path.isfile(hashed):
            return hashed
        flat = os.path.join(root, filename)
        if os.path.isfile(flat):
            return flat
        return hashed

    def existing_audio_by_id(self, folder):
        """Walk hash trees (or the whole folder) and map image_id -> basename."""
        lists = self.walk_audio_filenames_by_id(folder)
        return {iid: pick_audio_filename(names) for iid, names in lists.items()}

    def hash_dir_roots(self, folder):
        """Top-level 0-9/A-F hash directories under folder."""
        roots = []
        for top in self.HASH_TOP:
            path = os.path.join(folder, top)
            if os.path.isdir(path):
                roots.append(path)
        return roots

    def walk_audio_filenames_by_id(self, folder):
        """image_id -> [basenames] from hash folders, or a full walk if none exist."""
        by_id = {}
        if not os.path.isdir(folder):
            return by_id
        roots = self.hash_dir_roots(folder)
        if not roots:
            roots = [folder]
        n_files = 0
        for root in roots:
            for walk_root, _dirs, files in os.walk(root):
                if "_x" in walk_root.split(os.sep):
                    continue
                for fname in files:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in self.AUDIO_EXTS:
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

    def audio_on_disk(self, filename):
        filename = _clean_filename(filename)
        if not filename:
            return False
        path = self.resolve_audio_path(filename)
        return os.path.isfile(path)

    def filenames_from_metas_audio(self):
        if self._filenames_from_csv is None:
            self._filenames_from_csv = load_filenames_from_metas_audio(self.metas_audio_csv)
        return self._filenames_from_csv

    def walked_audio_by_id(self):
        """Walk hash folders once per process; keep every clip per image_id."""
        if self._walked_audio_by_id is None:
            root = self.sound_dir()
            print(f"Scraping hash-folder audio in {root} …")
            self._walked_audio_all = self.walk_audio_filenames_by_id(root)
            self._walked_audio_by_id = {
                iid: pick_audio_filename(names) for iid, names in self._walked_audio_all.items()
            }
            print(f"Walk found {len(self._walked_audio_by_id)} image_id(s) with audio")
        return self._walked_audio_by_id

    def metas_audio_fieldnames(self):
        """Header of metas_audio.csv, guaranteeing a filename column."""
        if self._metas_audio_fieldnames is not None:
            return self._metas_audio_fieldnames
        names = list(self.METAS_AUDIO_COLUMNS)
        if os.path.isfile(self.metas_audio_csv) and os.path.getsize(self.metas_audio_csv) > 0:
            with open(self.metas_audio_csv, "r", encoding="utf-8-sig", newline="") as f:
                header = list(csv.DictReader(f).fieldnames or [])
            if header:
                names = header
        if "filename" not in names:
            names.append("filename")
        self._metas_audio_fieldnames = names
        return names

    def append_metas_audio_rows(self, rows):
        """Append scraped cluster rows (with filename) to metas_audio.csv."""
        if not rows:
            return 0
        fieldnames = self.metas_audio_fieldnames()
        path = self.metas_audio_csv
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

    def append_scraped_cluster_rows(self, df, scraped):
        """Write metas.csv data + scraped filename for ids not yet in metas_audio.csv."""
        if not scraped:
            return 0
        csv_names = self.filenames_from_metas_audio()
        keys = df["image_id"].map(image_id_key)
        fieldnames = self.metas_audio_fieldnames()
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
        written = self.append_metas_audio_rows(rows)
        if self._filenames_from_csv is not None:
            self._filenames_from_csv.update(csv_names)
        return written

    def index_audio_for_topic(self, df):
        """Resolve audio for this cluster's ids; scrape hash folders; backfill metas_audio.csv.

        Returns (existing, still_missing). existing maps image_id -> audio basename.
        Ids found on disk but missing from metas_audio.csv are appended with the
        cluster metas.csv fields plus the scraped filename.
        """
        csv_names = self.filenames_from_metas_audio()
        walked = self.walked_audio_by_id()
        df_ids = {k for k in (image_id_key(v) for v in df["image_id"]) if k is not None}
        existing = {}
        still_missing = []
        scraped = []
        csv_ok = 0
        csv_stale = 0
        for iid in df_ids:
            csv_name = _clean_filename(csv_names.get(iid))
            if csv_name and self.audio_on_disk(csv_name):
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

        appended = self.append_scraped_cluster_rows(df, scraped)
        print(f"  metas_audio.csv hits on disk: {csv_ok}; "
              f"csv filename missing on disk: {csv_stale}; "
              f"scrape filled: {len(scraped)}; still missing: {len(still_missing)}")
        if appended:
            print(f"  appended {appended} row(s) to {self.metas_audio_csv}")
        return existing, still_missing

    def collect_missing_id_rows(self, df, missing_ids, cluster):
        """Append metas rows for ids with no audio to missing_ids.csv immediately."""
        if not missing_ids:
            return 0
        missing_set = set(missing_ids)
        keys = df["image_id"].map(image_id_key)
        rows = df[keys.isin(missing_set)].copy()
        if rows.empty:
            return 0
        rows.insert(0, "cluster", cluster)
        path = self.missing_ids_csv
        exists = os.path.isfile(path) and os.path.getsize(path) > 0
        if exists:
            _ensure_csv_trailing_newline(path)
            if self.missing_ids_fieldnames is None:
                with open(path, "r", encoding="utf-8-sig", newline="") as f:
                    self.missing_ids_fieldnames = list(csv.DictReader(f).fieldnames or [])
            fieldnames = self.missing_ids_fieldnames
            for col in fieldnames:
                if col not in rows.columns:
                    rows[col] = ""
            rows = rows[fieldnames]
        else:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            self.missing_ids_fieldnames = list(rows.columns)
        rows.to_csv(path, mode="a", header=not exists, index=False)
        n = len(rows)
        self.missing_ids_written += n
        print(f"  appended {n} missing-id row(s) to {path}")
        return n

    def write_missing_ids_csv(self, path=None):
        """Summarize missing_ids.csv appends for this run."""
        path = path or self.missing_ids_csv
        if self.missing_ids_written == 0:
            print(f"No missing ids appended to {path}")
            return
        print(f"Appended {self.missing_ids_written} missing-id row(s) this run to {path}")


class SurroundMixer:
    """Mix quiet / mid / loud clips into a quad or stereo soundtrack."""
    TARGET_SAMPLE_RATE = 24000
    CHUNK_SIZE = 500  # Adjust this value based on your system's capabilities
    OFFSET_DICT = {
        23: 0.075, # T23
        32: 0.0697, # T32
        37: 0.0755, # T37
    }
    WPS = 4
    SCALE_EXPONENT = 0  # exponent for scale_volume_exp; 0 = linear, 1 = cubic
    VOLUME_MIN = 0
    VOLUME_MAX = .8
    FIT_VOL_MIN = .1
    FIT_VOL_MAX = 1
    FADEOUT = 7
    FADE_TIME = 1
    QUIET = .5
    QUIET_VOL_MIN = .02
    QUIET_VOL_MAX = .08
    QUIET_PAD_FADE_IN = 3.0  # seconds to crossfade the tail pad into existing quiet
    QUIET_PAD_FADEOUT = 15   # per-clip fadeout, same default as scale_volume
    LOUD_ALOWED = 2
    LOUD_RESET = 7
    SPEAKER_NAMES = ("FL", "FR", "BL", "BR")
    STEREO_SPEAKER_NAMES = ("L", "R")
    CYCLE = (0, 1, 3, 2)  # FL, FR, BR, BL
    ANCHOR_RANGE_MID = [0.50, 0.75]
    ANCHOR_RANGE_LOUD = [0.40, 0.70]
    STEREO_QUIET_RANGE = (0.30, 0.70)
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

    def __init__(self, input_dir, batch_mode=True, batch_folder=None, batch_clusters=None,
                 topic=0, key_topics=None, sound_folder=".", io=None):
        self.input_dir = input_dir
        self.batch_mode = batch_mode
        self.batch_folder = batch_folder
        self.batch_clusters = list(batch_clusters or [])
        self.topic = topic
        self.key_topics = list(key_topics) if key_topics is not None else [0, 3, 15, 45]
        self.audio = AudioIndex(input_dir, sound_folder=sound_folder, io=io)
        self.csv_file = f"metas_{topic}.csv"
        self.offset = self.OFFSET_DICT.get(topic, 0.0743)
        self.loud_counter = []
        self.channel_counter = 0
        self.fake_loud = False
        self.existing_files = {}
        self.output_mode = "quad"
        self.n_channels = 4

    def choose_output_mode(self):
        """Ask once per run. Quad stays 4-channel; stereo places in L/R."""
        title = "Choose soundtrack output:"
        options = [
            "quad (4-channel FL FR BL BR)",
            "stereo",
        ]
        _option, index = pick(options, title)
        if index == 1:
            self.output_mode = "stereo"
            self.n_channels = 2
        else:
            self.output_mode = "quad"
            self.n_channels = 4
        print(f"Output mode: {self.output_mode} ({self.n_channels} channels)")

    def stereo_gains(self, tier):
        """L/R weights that sum to 1.

        Quiet keeps each side inside self.STEREO_QUIET_RANGE so a clip cannot hard-pan.
        Mid and loud give one side the anchor share and the other side the rest.
        Loud alternates the anchor side with self.channel_counter, which ticks once
        per above-quiet clip just before this is called.
        """
        if tier == "quiet":
            lo, hi = self.STEREO_QUIET_RANGE
            left = float(np.random.uniform(lo, hi))
            return np.array([left, 1.0 - left]), None

        if tier == "mid":
            side = int(np.random.randint(0, 2))
            lo_hi = self.ANCHOR_RANGE_MID
        else:
            side = (self.channel_counter - 1) % 2 if self.channel_counter else 0
            lo_hi = self.ANCHOR_RANGE_LOUD
        share = _anchor_share(lo_hi)
        gains = np.zeros(2)
        gains[side] = share
        gains[1 - side] = 1.0 - share
        return gains, (self.STEREO_SPEAKER_NAMES[side], share, lo_hi)

    def spatial_gains(self, tier):
        """Weights that sum to 1 over the chosen speakers.

        Quad: FL, FR, BL, BR. Stereo: L, R via stereo_gains.
        Returns (gains, anchor_info). anchor_info is None for quiet, else
        (speaker_name, share, [lo, hi]).
        """
        if self.output_mode == "stereo":
            return self.stereo_gains(tier)
        gains = np.zeros(self.n_channels)
        start = np.random.randint(0, 4)
        if tier == "quiet":
            idxs = [self.CYCLE[start], self.CYCLE[(start + 1) % 4]]
            weights = random_normalized_weights(len(idxs))
            for idx, weight in zip(idxs, weights):
                gains[idx] = weight
            return gains, None

        if tier == "mid":
            anchor = self.CYCLE[start]
            others = [self.CYCLE[(start - 1) % 4], self.CYCLE[(start + 1) % 4]]
            lo_hi = self.ANCHOR_RANGE_MID
            share = _anchor_share(lo_hi)
            remainder = random_normalized_weights(2) * (1.0 - share)
            gains[anchor] = share
            for idx, weight in zip(others, remainder):
                gains[idx] = weight
            return gains, (self.SPEAKER_NAMES[anchor], share, lo_hi)

        # loud: any speaker as anchor; remainder split among the other three
        anchor = start
        others = [i for i in range(self.n_channels) if i != anchor]
        lo_hi = self.ANCHOR_RANGE_LOUD
        share = _anchor_share(lo_hi)
        remainder = random_normalized_weights(3) * (1.0 - share)
        gains[anchor] = share
        for idx, weight in zip(others, remainder):
            gains[idx] = weight
        return gains, (self.SPEAKER_NAMES[anchor], share, lo_hi)

    def keys_for_search(self):
        """Union of KEYS stems for every id in KEY_TOPICS, first occurrence kept."""
        stems = []
        seen = set()
        for t in self.key_topics:
            for key in self.KEYS.get(t, []):
                if key not in seen:
                    seen.add(key)
                    stems.append(key)
        return stems

    def search_for_keys(self, row):
        # search the first three words of the description for each key in self.KEYS
        # if any of the keys are found, set the volume to 1
        # if not, set the volume to 0.5
        if pd.isna(row['description']): return [],0

        # found = False
        found_list=[]
        desc_split=row['description'].lower().split(" ")
        desc_count=len(desc_split)
        active_keys = self.keys_for_search()
        for index,word in enumerate(desc_split):
            for key in active_keys:
                if key in word:
                    print(" ---- ", key, "found in", word, row['description'],row['image_id'])
                    found_list.append(index)
                    break
        if len(found_list)==0:
            print("No keys found in", row['description'],"for topic models", self.key_topics)
        return found_list,desc_count

    def spatial_tier(self, row):
        """Match scale_volume branches: quiet / loud(keys) / mid."""
        volume_fit = float(row["topic_fit"])
        if volume_fit < self.QUIET:
            return "quiet"
        key_index, _ = self.search_for_keys(row)
        if len(key_index) > 0:
            return "loud"
        return "mid"

    def scale_volume(self, row, cycler, audio_data, sample_rate):
        def is_bark_loud(row):
            # image_id = float(row['topic_fit'])  # Using topic_fit as the volume level 
            image_id = row['image_id']  # Using topic_fit as the volume level
            path = self.existing_files.get(image_id_key(image_id))
            # if path containts meta, return True
            # TEMP CHANGE (was: if "bark_v5" in path: return True)
            if path is None: return False
            if "bark_v5" in path: return True
            else: return False

        volume_fit = float(row['topic_fit'])  # Using topic_fit as the volume level 
        # defaults
        fadein = 0
        fadeout = 15

        # search_for_keys to see where the matching keys are
        key_index,desc_count=self.search_for_keys(row)

        if volume_fit < self.QUIET:
            # vol = scale_volume_exp(volume_fit, 3)
            vol = scale_volume_linear(volume_fit, self.QUIET_VOL_MIN, self.QUIET_VOL_MAX)*cycler[0]
            # vol = .001
        elif len(key_index)>0:
            # if keys are found, set the volume and fade in out based on the keys found
            fadein,fadeout=calculate_fades(key_index,desc_count, audio_data, sample_rate)
            vol = scale_volume_exp(volume_fit,self.SCALE_EXPONENT)*1
            print(key_index)
            # start,end=key_index[0],key_index[-1]
            # vol =0
            # if vol < .5: vol = .001
            if vol < self.QUIET: 
                if vol > self.QUIET/2:
                    # vol = vol - len(self.loud_counter)*.1
                    if len(self.loud_counter) == 0:
                        if not self.fake_loud:
                            # trying to only trigger this once per self.loud_counter cycle
                            self.fake_loud = True
                            print("ffffffff    Fake loud set")

                            # if there are no loud files, scale the volume between .4 and .8
                            # to fill silence
                            vol = scale_volume_linear(volume_fit,0 ,.8)
                        else:
                            vol = (vol*.35) *cycler[1]
                            # vol = vol / (len(self.loud_counter)*.5+1)
                            # vol = .001
                    else:
                        # reduce the volume of the audio based on the number of loud files
                        vol = vol / (len(self.loud_counter)*.5+1)
                    if np.max(np.abs(audio_data)) > .8: vol = vol/3
                    # if vol > .8: vol = .8
                    # vol = .001
                else:
                    vol = (vol*.45) *cycler[1]
                    # vol = .001
            elif is_bark_loud(row):
                if np.max(np.abs(audio_data)) > self.QUIET: vol = vol/3
            # else: vol = .001
        else:
            vol = scale_volume_linear(volume_fit, .04,.15)*cycler[1]
            # if vol > .1: vol = .1
            # vol = vol*cycler[1]
            print("cylcerl vol",vol)
            # vol = .001
        return vol, fadeout,fadein

    def build_quiet_background(self, quiet_files, total_duration, start_time=0.0, offset=None,
                               fade_in=None):
        """Overlapping quiet bed from start_time to total_duration on the OFFSET grid.

        Matches the main mixer: a new clip every OFFSET seconds, quiet-tier volume
        and two-adjacent-speaker placement. Starts *fade_in* seconds before
        start_time so the pad crossfades as the original quiet layer dies out.
        The file pool is shuffled on every pass so the tail is not a literal repeat.
        """
        if not quiet_files or total_duration <= 0:
            return None

        offset = self.offset if offset is None else offset
        if fade_in is None:
            fade_in = self.QUIET_PAD_FADE_IN
        clips = _load_quiet_clips(quiet_files)
        if not clips:
            print("build_quiet_background: no readable quiet files, aborting pad")
            return None

        pad_start = max(0.0, start_time - fade_in)
        if pad_start >= total_duration:
            return None

        total_samples = int(total_duration * self.TARGET_SAMPLE_RATE)
        background = np.zeros((total_samples, self.n_channels))

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

            clip_vol = np.random.uniform(self.QUIET_VOL_MIN, self.QUIET_VOL_MAX)
            mono = _fadeout_mono(clip * clip_vol, self.QUIET_PAD_FADEOUT)
            gains, _anchor = self.spatial_gains("quiet")
            audio = apply_quad_gains(mono, gains)

            start_sample = int(t * self.TARGET_SAMPLE_RATE)
            if start_sample >= total_samples:
                break
            end_sample = min(start_sample + len(audio), total_samples)
            n_copy = end_sample - start_sample
            if n_copy > 0:
                background[start_sample:end_sample] += audio[:n_copy]
                n_placed += 1
            t += offset

        fade_begin = int(pad_start * self.TARGET_SAMPLE_RATE)
        fade_samples = int(fade_in * self.TARGET_SAMPLE_RATE)
        fade_end = min(fade_begin + fade_samples, total_samples)
        n_fade = fade_end - fade_begin
        if n_fade > 1:
            fade_curve = np.power(np.linspace(0.0, 1.0, n_fade), 2)
            background[fade_begin:fade_end] *= fade_curve[:, np.newaxis]

        print(f"build_quiet_background: placed {n_placed} overlapping clips "
              f"from {pad_start:.1f}s to {total_duration:.1f}s "
              f"(offset={offset:.4f}s, {n} unique files)")
        return background

    def process_audio_chunk(self, chunk_df, start_index, chunk_index):
        channel_data = [[] for _ in range(self.n_channels)]
        quiet_files_used = []   # paths of files placed in the quiet tier
        quiet_max_end_time = 0  # latest end time seen for a quiet-tier clip
        max_end_time = 0
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

            # input_path = os.path.join(self.input_dir, row['out_name'])
            # input_path = row['out_name']

            # if os.path.exists(input_path):
            #     good_files.append(input_path)
            # elif 
            print("Row:", row)
            image_id = row['image_id']
            description = row['description']
            # print("Image ID:", image_id)
            # if str(image_id) in self.existing_files.keys(): 
            #     print("^^^^^^^^ Image_id in existing files already ^^^^^^^^^^^^^^^",image_id)
            #     print("file path",self.existing_files.get(str(image_id)))
            # else:
            #     print(image_id,"^^^^^^ image_id not in existing files ^^^^^^^^^^^")
            #     print("existing files",self.existing_files)

            iid = image_id_key(image_id)
            if pd.notna(description) and iid is not None and iid in self.existing_files:
                input_file = self.existing_files.get(iid)
                print("Using existing file:", input_file)
            elif pd.notna(description) and image_id:
                if not self.existing_files:
                    print(f"Skipping image_id {image_id}: no existing files to fall back to")
                    continue
                input_file = np.random.choice(list(self.existing_files.values()))
                if row['topic_fit'] < .6:
                    print("unprocessed meta file")
                elif row['topic_fit'] > .75:
                    print("unprocessed openai file")
            elif pd.isna(description) and image_id:
                if not self.existing_files:
                    print(f"Skipping image_id {image_id} (NaN description): no existing files to fall back to")
                    continue
                input_file = np.random.choice(list(self.existing_files.values()))
                if row['topic_fit'] > self.QUIET:
                    row['topic_fit'] = row['topic_fit']/2
                print(f"is NaN assigned random file: {input_file} and topic_fit halved to {row['topic_fit']}")
            else:
                print("No good files found")
                continue

            input_path = self.audio.resolve_audio_path(input_file)


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
            # found = self.search_for_keys(row)

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
            volume_scale, fadeout,fadein = self.scale_volume(row, cycler, audio_data, sample_rate)
            audio_data_adjusted = audio_data * volume_scale
            # print(f"volume_fit:", volume_fit, "scaled_vol" ,volume_scale, "Pan:", pan, fadeout)

            # count the loud audio files
            # subtract self.offset from each value in the loud counter
            # do this each loop, regardless of the volume
            loud_delay_duration = 0
            loud_offset = 0

            if self.loud_counter and len(self.loud_counter) > 0:
                self.loud_counter = [x - self.offset for x in self.loud_counter]
                print("Loud counter:", len(self.loud_counter))
                print("Loud counter:", self.loud_counter)
                # if any value in the loud counter is less than 0, remove it
                self.loud_counter = [x for x in self.loud_counter if x > 0]
                print("Loud counter:", len(self.loud_counter))
                if self.fake_loud and len(self.loud_counter) == 0:
                    # if self.loud_counter cycle is complete, reset self.fake_loud
                    self.fake_loud = False
                    print("rrrrrrrrrrrrr    Fake loud reset")
                if len(self.loud_counter) > self.LOUD_ALOWED*self.LOUD_RESET:
                    # reset the loud counter if it gets too long
                    self.loud_counter = []
            if self.loud_counter and len(self.loud_counter) >= self.LOUD_ALOWED and volume_scale > self.QUIET:
                # audio_data_adjusted = audio_data_adjusted* (1/len(self.loud_counter))


                if len(self.loud_counter) > self.LOUD_ALOWED*2:
                        # if there is a backlog of loud files, reduce the volume and play normal speed
                        # otherwise, the track will be 3x long, with the last 2x all the loud files
                    # loud_divisor = min(len(self.loud_counter), 10)
                    audio_data_adjusted = audio_data_adjusted * (1 / (2 + len(self.loud_counter)))
                else:
                    print("TOO LOUD")
                    loud_delay_duration = 2* (len(self.loud_counter)-self.LOUD_ALOWED)
                    loud_offset = (max(self.loud_counter)/self.offset) +(loud_delay_duration/self.offset)
                    print("Loud offset:", loud_offset)
            if volume_scale > self.QUIET:
                self.loud_counter.append(len(audio_data)/sample_rate)
                self.channel_counter += 1

            # Apply fadeout to the audio data
            apply_fadeout(audio_data_adjusted, sample_rate, fadeout)
            ################
            # Apply fadein to the audio data
            if fadein>0:apply_fadein(audio_data_adjusted, sample_rate, fadein)
            ####################
            tier = self.spatial_tier(row)
            gains, anchor_info = self.spatial_gains(tier)
            audio_data_adjusted = apply_quad_gains(audio_data_adjusted, gains)
            if anchor_info is not None:
                name, share, lo_hi = anchor_info
                print(f"Anchor {name}={share:.2f} range=[{lo_hi[0]:.2f}, {lo_hi[1]:.2f}] ({tier})")
            names = self.STEREO_SPEAKER_NAMES if len(gains) == 2 else self.SPEAKER_NAMES
            print(f"Spatial {tier}:", " ".join(
                f"{name}={g:.2f}" for name, g in zip(names, gains) if g > 1e-6
            ))

            # # Append audio data to respective lists
            # left_channel_data.append(audio_data_adjusted[:, 0])
            # right_channel_data.append(audio_data_adjusted[:, 1])
            repeat, last_description = test_repeat(description, last_description)
            # Calculate the start time for this audio clip
            # if repeat, then start at the same time as the last clip
            start_time = (start_index + i - repeat + loud_offset) * self.offset
            end_time = start_time + len(audio_data_adjusted) / self.TARGET_SAMPLE_RATE
            max_end_time = max(max_end_time, end_time)

            # track quiet-tier (lowest volume) files so we can loop them later if needed
            if float(row['topic_fit']) < self.QUIET:
                quiet_files_used.append(input_path)
                quiet_max_end_time = max(quiet_max_end_time, end_time)

            # Create arrays with the correct offset
            n_samples = int(np.ceil(end_time * self.TARGET_SAMPLE_RATE))
            placed = [np.zeros(n_samples) for _ in range(self.n_channels)]

            # Insert the audio data at the correct position
            start_sample = int(start_time * self.TARGET_SAMPLE_RATE)
            end_sample = min(start_sample + len(audio_data_adjusted), n_samples)
            n_copy = end_sample - start_sample
            for ch in range(self.n_channels):
                placed[ch][start_sample:end_sample] = audio_data_adjusted[:n_copy, ch]
                channel_data[ch].append(placed[ch])

        # If no audio was collected (all rows skipped), return silence
        if not channel_data[0]:
            print("process_audio_chunk: no audio collected for this chunk, returning silence")
            silence = np.zeros((self.TARGET_SAMPLE_RATE, self.n_channels))
            return silence, 0.0, quiet_files_used, quiet_max_end_time

        # Mix the audio data for the chunk
        max_length = max(len(data) for ch in channel_data for data in ch)
        mixed_audio = np.zeros((max_length, self.n_channels))

        n_clips = len(channel_data[0])
        for i in range(n_clips):
            for ch in range(self.n_channels):
                clip = channel_data[ch][i]
                mixed_audio[:len(clip), ch] += clip

        # Clear memory
        del channel_data
        gc.collect()

        # save the mixed audio to a file
        # output_file = os.path.join(self.input_dir, f"multitrack_mixdown_offset_{self.topic}_{chunk_index}.wav")
        # sf.write(output_file, mixed_audio, self.TARGET_SAMPLE_RATE, format='wav')

        return mixed_audio, max_end_time, quiet_files_used, quiet_max_end_time

    def resolve_batch_folder(self, folder_name):
        """Return an absolute path to the parent folder of cluster directories."""
        if os.path.isabs(folder_name):
            return folder_name
        return os.path.join(self.input_dir, folder_name)

    def list_cluster_jobs(self, parent, names=None):
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
                print(f"Skipping {label}: no {AudioIndex.METAS_CSV_NAME} in {folder}")
                continue
            jobs.append((label, csv_path))
        return jobs

    def run_topic(self, topic, csv_path=None):
        """Process a single topic/cluster and write its output file."""

        # configure globals for this topic
        self.topic = topic
        if csv_path is None:
            self.csv_file = f"metas_{self.topic}.csv"
            csv_path = os.path.join(self.input_dir, "audioproduction", self.csv_file)
        else:
            self.csv_file = os.path.basename(csv_path)
        self.offset = self.OFFSET_DICT.get(self.topic, 0.0743)

        # reset stateful globals so each topic starts clean
        self.loud_counter = []
        self.channel_counter = 0
        self.fake_loud = False

        suffix = "stereo" if self.output_mode == "stereo" else "quad"
        output_path = os.path.join(self.input_dir, f"multitrack_mixdown_offset_{self.topic}_{suffix}.wav")

        topic_t0 = time.time()
        print(f"\n{'='*60}")
        print(f"[Topic {self.topic}] Starting — CSV: {csv_path}  OFFSET: {self.offset}  OUTPUT: {self.output_mode}")
        missing_key_topics = [t for t in self.key_topics if t not in self.KEYS]
        if missing_key_topics:
            print(f"[Topic {self.topic}] WARNING: KEY_TOPICS not in KEYS dict: {missing_key_topics}")
        print(f"[Topic {self.topic}] KEY_TOPICS {self.key_topics} → {self.keys_for_search()}")
        print(f"{'='*60}")

        df = read_metas_csv(csv_path)

        print(f"[Topic {self.topic}] Resolving audio via metas_audio.csv + hash-folder scrape")
        self.existing_files, missing_ids = self.audio.index_audio_for_topic(df)
        self.audio.collect_missing_id_rows(df, missing_ids, self.topic)
        print(f"[Topic {self.topic}] Existing files after INTERSECT:", len(self.existing_files))
        for k, v in list(self.existing_files.items())[:5]:
            print(f"  self.existing_files key: {repr(k)}  ->  {self.audio.resolve_audio_path(v)}")

        if os.path.exists(output_path):
            print(f"[Topic {self.topic}] Output already exists, skipping: {output_path}")
            return None

        combined_audio = None
        start_index = 0
        all_quiet_files = []      # accumulate quiet-tier file paths across all chunks
        quiet_coverage_end = 0.0  # track the furthest end time of any quiet-tier clip

        chunks = read_metas_csv(csv_path, chunksize=self.CHUNK_SIZE)
        for chunk_index, chunk in enumerate(chunks):
            chunk_audio, chunk_end_time, chunk_quiet_files, chunk_quiet_end = self.process_audio_chunk(chunk, start_index, chunk_index)
            print(f"[Topic {self.topic}] Chunk audio length/sample:", len(chunk_audio)/self.TARGET_SAMPLE_RATE, "Chunk end time:", chunk_end_time)

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
        total_duration = len(combined_audio) / self.TARGET_SAMPLE_RATE
        print(f"[Topic {self.topic}] Quiet tier reached {quiet_coverage_end:.1f}s / {total_duration:.1f}s total")
        gap = total_duration - quiet_coverage_end
        if all_quiet_files and gap > self.offset:
            print(f"[Topic {self.topic}] Building overlapping quiet pad ({gap:.1f}s gap, "
                  f"{len(set(all_quiet_files))} unique files, offset={self.offset}s)…")
            quiet_bg = self.build_quiet_background(
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
                print(f"[Topic {self.topic}] Quiet pad mixed in ({len(quiet_bg)/self.TARGET_SAMPLE_RATE:.1f}s)")
        elif not all_quiet_files:
            print(f"[Topic {self.topic}] No quiet-tier files collected — skipping quiet pad")
        else:
            print(f"[Topic {self.topic}] Quiet tier already covers the track, skipping pad")

        print(f"[Topic {self.topic}] Combined audio shape before writing:", combined_audio.shape)
        print(f"[Topic {self.topic}] Writing to file:", output_path)
        sf.write(output_path, combined_audio, self.TARGET_SAMPLE_RATE, format='wav')
        if self.output_mode == "quad":
            tag_quad_wav(output_path)
        elapsed = time.time() - topic_t0
        print(f"[Topic {self.topic}] Time to process output file: {elapsed:.1f}s")
        del combined_audio
        gc.collect()
        return elapsed

    def run(self):
        self.audio.missing_ids_written = 0
        self.audio.missing_ids_fieldnames = None
        self.choose_output_mode()
        self.audio.filenames_from_metas_audio()
        self.audio.walked_audio_by_id()
        elapsed_times = []
        if self.batch_mode:
            batch_dir = self.resolve_batch_folder(self.batch_folder)
            jobs = self.list_cluster_jobs(batch_dir, self.batch_clusters)
            print(f"Batch mode ON — found {len(jobs)} cluster folder(s) in {batch_dir}")
            if not jobs:
                print(f"No cluster folders with {AudioIndex.METAS_CSV_NAME} found.")
                return
            for topic, csv_path in jobs:
                elapsed = self.run_topic(topic, csv_path=csv_path)
                if elapsed is not None:
                    elapsed_times.append(elapsed)
            print("\nBatch complete.")
        else:
            elapsed = self.run_topic(self.topic)
            if elapsed is not None:
                elapsed_times.append(elapsed)
        self.audio.write_missing_ids_csv()
        if elapsed_times:
            avg = sum(elapsed_times) / len(elapsed_times)
            print(f"Average processing time per cluster: {avg:.1f}s "
                  f"({len(elapsed_times)} cluster(s))")
        else:
            print("Average processing time per cluster: n/a (no clusters processed)")

def check_fade_length(fade_length, audio_data_adjusted, sample_rate=SurroundMixer.TARGET_SAMPLE_RATE):
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
    if sample_rate != SurroundMixer.TARGET_SAMPLE_RATE:
        # Resample the audio to 24000 Hz
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=SurroundMixer.TARGET_SAMPLE_RATE)
    return audio_data, sample_rate

def scale_volume_exp(volume_fit, exponent=3):
    exp_vol = (volume_fit - SurroundMixer.FIT_VOL_MIN)**exponent / (SurroundMixer.FIT_VOL_MAX  - SurroundMixer.FIT_VOL_MIN)**exponent * (SurroundMixer.VOLUME_MAX - SurroundMixer.VOLUME_MIN) + SurroundMixer.VOLUME_MIN
    return exp_vol

def scale_volume_linear(volume_fit, min_out = SurroundMixer.VOLUME_MIN, max_out = SurroundMixer.VOLUME_MAX):
    linear_vol = (volume_fit - SurroundMixer.FIT_VOL_MIN) / (SurroundMixer.FIT_VOL_MAX  - SurroundMixer.FIT_VOL_MIN) * (max_out - min_out) + min_out
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

def to_mono(audio):
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)

def random_normalized_weights(n):
    w = np.random.random(n)
    s = w.sum()
    if s <= 0:
        return np.ones(n) / n
    return w / s

def _anchor_share(range_lo_hi):
    lo, hi = range_lo_hi
    return float(np.random.uniform(lo, hi))

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

def _load_quiet_clips(quiet_files):
    """Load unique quiet files to mono at SurroundMixer.TARGET_SAMPLE_RATE."""
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

def _fadeout_mono(audio, duration, sample_rate=SurroundMixer.TARGET_SAMPLE_RATE):
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

def merge_audio(combined_audio, chunk_audio_without_silence):
    # Assuming sample_rate is defined
    # sample_rate = SurroundMixer.TARGET_SAMPLE_RATE  # Example sample rate, replace with your actual sample rate
    overlap_duration = 10  # Duration in seconds
    overlap_samples = SurroundMixer.TARGET_SAMPLE_RATE * overlap_duration

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
    # sf.write(str(len(c ombined_audio))+"combined_audio.wav", combined_audio, SurroundMixer.TARGET_SAMPLE_RATE, format='wav')
    return combined_audio

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
    return {"header": None, "names": AudioIndex.METAS_COLUMNS}

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
    return os.path.join(folder, AudioIndex.METAS_CSV_NAME)

def resolve_cluster_folder(entry, parent):
    """Resolve a BATCH_CLUSTERS entry to an absolute cluster folder path."""
    if os.path.isabs(entry):
        return entry
    return os.path.join(parent, entry)

def main():
    mixer = SurroundMixer(
        INPUT,
        batch_mode=BATCH_MODE,
        batch_folder=BATCH_FOLDER_NAME,
        batch_clusters=BATCH_CLUSTERS,
        topic=TOPIC,
        key_topics=KEY_TOPICS,
        sound_folder=SOUND_FOLDER,
        io=io,
    )
    mixer.run()


if __name__ == "__main__":
    main()
