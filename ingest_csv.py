from multiprocessing import Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception
import csv
import os
import hashlib
import cv2
import math
import pickle
import sys # can delete for production
import pathlib
import re
import traceback
from collections import Counter
from retrying import retry

import numpy as np
import pandas as pd
from pyinflect import getInflection

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
# my ORM
from my_declarative_base import Base, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, Images, ImagesEthnicity, ImagesKeywords

from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, String, VARCHAR, Float, ForeignKey, Date, update, insert, select, PrimaryKeyConstraint, UniqueConstraint
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool
from sqlalchemy.dialects.mysql import insert

#mine
from mp_db_io import DataIO

sig = '''

_)                       |                     
 | __ \   _` |  _ \  __| __|   __|  __|\ \   / 
 | |   | (   |  __/\__ \ |    (   \__ \ \ \ /  
_|_|  _|\__, |\___|____/\__| \___|____/  \_/   
        |___/                                  
'''


######## Michael's Credentials ########
# platform specific credentials
io = DataIO()
db = io.db
# overriding DB for testing
io.db["name"] = "ministock"
ROOT = io.ROOT 
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES
#######################################

# starting shutter here: 52344682

# INGEST_ROOT = "/Users/michaelmandiberg/Documents/projects-active/facemap_production"
# INGEST_FOLDER = os.path.join(INGEST_ROOT, "adobe_csv_4ingest/")
# CSV_IN_PATH = os.path.join(INGEST_FOLDER, "unique_lines_B_nogender.csv")
INGEST_FOLDER = "/Volumes/SSD4/CSVs_to_ingest/shutterstockCSVs"
CSV_IN_PATH = os.path.join(INGEST_FOLDER, "shutttersock_big4.output.csv")
KEYWORD_PATH = os.path.join(INGEST_FOLDER, "Keywords_202305150950.csv")
LOCATION_PATH = os.path.join(INGEST_FOLDER, "Location_202308041952.csv")
CSV_NOKEYS_PATH = os.path.join(INGEST_FOLDER, "CSV_NOKEYS.csv")
CSV_IMAGEKEYS_PATH = os.path.join(INGEST_FOLDER, "CSV_IMAGEKEYS.csv")
# NEWIMAGES_FOLDER_NAME = 'images_pexels'
CSV_COUNTOUT_PATH = os.path.join(INGEST_FOLDER, "countout.csv")
CSV_NOLOC_PATH = os.path.join(INGEST_FOLDER, "CSV_NOLOC.csv")
CSV_BLANKLOC_PATH = os.path.join(INGEST_FOLDER, "CSV_BLANKLOC_PATH.csv")
CSV_ETH2_PATH = os.path.join(INGEST_FOLDER, "CSV_ETH2.csv")

SEARCH_KEYS_FOR_LOC = True

# key2key = {"person":"people", "kid":"child","affection":"Affectionate", "baby":"Baby - Human Age", "beautiful":"Beautiful People", "pretty":"Beautiful People", "blur":"Blurred Motion", "casual":"Casual Clothing", "children":"Child", "kids":"Child", "couple":"Couple - Relationship", "adorable":"Cute", "room":"Domestic Room", "focus":"Focus - Concept", "happy":"Happiness", "at home":"Home Interior", "home":"Home Interior", "face":"Human Face", "hands":"Human Hand", "landscape":"Landscape - Scenery", "outfit":"Landscape - Scenery", "leisure":"Leisure Activity", "love":"Love - Emotion", "guy":"Men", "motherhood":"Mother", "parenthood":"Parent", "positive":"Positive Emotion", "recreation":"Recreational Pursuit", "little":"Small", "studio shoot":"Studio Shot", "together":"Togetherness", "vertical shot":"Vertical", "lady":"women", "young":"Young Adult"}
loc2loc = {"niue":"Niue Island", "east timor":"timor-leste"}
key2key = {"person":"people", "isolated on white":"plain white background", "kid":"child","affection":"affectionate", "baby":"baby - human age", "beautiful":"beautiful people", "pretty":"beautiful people", "blur":"blurred motion", "casual":"casual clothing", "children":"child", "kids":"child", "couple":"couple - relationship", "adorable":"cute", "room":"domestic room", "focus":"focus - concept", "happy":"happiness", "at home":"home interior", "home":"home interior", "face":"human face", "hands":"human hand", "landscape":"landscape - scenery", "outfit":"landscape - scenery", "leisure":"leisure activity", "love":"love - emotion", "guy":"men", "motherhood":"mother", "parenthood":"parent", "positive":"positive emotion", "recreation":"recreational pursuit", "little":"small", "studio shoot":"studio shot", "together":"togetherness", "vertical shot":"vertical", "lady":"women", "young":"young adult", "light":"light - natural phenomenon", "trees":"tree", "disabled":"disability", "landline":"phone", "tradesman":"worker", "apprentice":"work", "arbeit":"work", "wheel-chair":"wheelchair", "treatments":"treatment", "transports":"transportation", "thoughtfully":"thoughtful", "technologies":"technology", "piscine":"swim", "astonished":"surprise", "surgeons":"surgeon", "sommer":"summer", "suffering":"suffer", "studentin":"student", "stressful":"stressed", "smoothies":"smoothie", "smilling":"smiling", "kleines":"small", "sleeps":"sleeping", "dealership":"sales", "salads":"salad", "ressources":"resources", "relaxes":"relaxed", "presentations":"presentation", "phones":"phone", "telefon":"phone", "telefoniert":"phone", "patients":"patient", "papier":"paper", "painful":"pain", "offended":"offend", "occupations":"occupation", "muscled":"muscles", "motivated":"motivation", "pinup":"model", "pin-up":"model", "meetings":"meeting", "massages":"massage", "kleiner":"little", "(lawyer)":"lawyer", "kitchens":"kitchen", "injections":"injection", "hospitals":"hospital", "zuhause":"home", "happily":"happy", "joyfully":"happy", "overjoyed":"happiness", "rejoices":"happiness", "handshaking":"handshake", "groups":"group", "full-length":"Full Length", "blumen":"flowers", "florists":"florist", "panic":"fear", "fell":"fall", "equipements":"equipement", "enthusiastic":"enthusiasm", "osteopathy":"doctor", "disgusted":"disgust", "schreibtisch":"desk", "dances":"dancing", "crowds":"crowd", "robber":"criminal", "copyspace":"Copy Space", "misunderstandings":"confusion", "confidently":"confidence", "concerts":"concert", "climbs":"climb", "celebrations":"celebration", "caught":"catch", "casually":"casual", "motorsports":"car", "banker":"Business Person", "supervisor":"boss", "executives":"boss", "bedrooms":"bedroom", "beautifull":"beautiful", "beaches":"beach", "bathrooms":"bathroom", "backgroud":"background", "attraktive":"attractive", "sportwear":"athletic", "sportliche":"athletic", "addicted":"addiction", "alcoholism":"addiction", "enjoy":"enjoyment"}
skip_keys = ["other"]

gender_dict = {"men":1,"man":1,"male":1,"males":1,"his":1,"him":1,"businessman":1,"businessmen":1,"father":1, "men's":1, "himself":1, "homme":1, "hombre":1, "(man)":1, "-women men -children":1, "-women -men -children":2, "none":2, "oldmen":3, "grandfather":3,"oldwomen":4, "grandmother":4, "nonbinary":5, "other":6, "trans":7, 
        "women":8,"woman":8,"female":8,"females":8, "hers":8, "her":8, "businesswoman":8, "businesswomen":8, "mother":8, "frauen":8, "mujer":8, "haaren":8, "frau":8, "woman-doctor":8, "maiden":8, "hausfrau":8, "women -men -children":8, "youngmen":9, "boy":9, "boys":9, "jungen":9, "youngwomen":10,"girl":10, "girls":10, "ragazza":10, "schoolgirls":8,}
gender_dict_istock = {"Mid Adult Men":1, "Only Mid Adult Men":1, "One Mid Adult Man Only":1, "Only Men":1, "One Man Only":1, "Senior Men":3, "Only Senior Men":3, "One Senior Man Only":3, "Mature Men":3, "Only Mature Men":3, "One Mature Man Only":3, "Mature Women":4, "Only Mature Women":4, "One Mature Woman Only":4, "Senior Women":4, "Only Senior Women":4, "One Senior Woman Only":4, "Mid Adult Women":8, "Only Mid Adult Women":8, "One Mid Adult Woman Only":8, "Only Women":8, "One Woman Only":8, "Young Men":9, "Only Young Men":9, "One Young Man Only":9, "Teenage Boys":9, "Only Teenage Boys":9, "One Teenage Boy Only":9, "Only Boys":9, "One Boy Only":9, "Baby Boys":9, "Only Baby Boys":9, "One Baby Boy Only":9, "Young Women":10, "Only Young Women":10, "One Young Woman Only":10, "Teenage Girls":10, "Only Teenage Girls":10, "One Teenage Girl Only":10, "Only Girls":10, "One Girl Only":10, "Baby Girls":10, "Only Baby Girls":10, "One Baby Girl Only":10}
gender_dict_sex = {"Mid Adult Male":1, "Only Mid Adult Male":1, "One Mid Adult Man Only":1, "Only Male":1, "One Man Only":1, "Senior Male":3, "Only Senior Male":3, "One Senior Man Only":3, "Mature Male":3, "Only Mature Male":3, "One Mature Male Only":3, "Mature Female":4, "Only Mature Female":4, "One Mature Woman Only":4, "Senior Female":4, "Only Senior Female":4, "One Senior Woman Only":4, "Mid Adult Female":8, "Only Mid Adult Female":8, "One Mid Adult Woman Only":8, "Only Female":8, "One Woman Only":8, "Young Male":9, "Only Young Male":9, "One Young Man Only":9, "Young Female":10, "Only Young Female":10, "One Young Female Only":10}
gender_dict_sexplural = {"Mid Adult Males":1, "Only Mid Adult Males":1, "One Mid Adult Man Only":1, "Only Males":1, "Senior Males":3, "Only Senior Males":3,  "Mature Males":3, "Only Mature Males":3,  "Mature Females":4, "Only Mature Females":4, "Senior Females":4, "Only Senior Females":4, "Mid Adult Females":8, "Only Mid Adult Females":8,  "Only Females":8, "One Woman Only":8, "Young Males":9, "Young Adult Males":9, "Young Adult Men":9, "Young Adult Man":9, "Only Young Males":9, "Young Females":10, "Young Ault Women":10, "Young Ault Woman":10, "Young Ault Female":10, "Young Ault Females":10, "Only Young Females":10}
gender_dict_shutter_secondary = {"3 year old girl":8, "3 years old girl":8, "4 year old girl":8, "5 year old girl":8, "7 year old girl":8, "9 year old girl":8, "acne girl":8, "active girl":8, "adorable girl":8, "adorable little girl":8, "african american girl":8, "african girl":8, "alone girl":8, "alternative girl":8, "amazon girl":8, "anime girl":8, "anxious girl":8, "apocalypse girl":8, "asian baby girl":8, "asian girl":8, "asian girl face":8, "asian girl isolated":8, "asian girls":8, "asian hijab girl":8, "attractive girl":8, "attractive teenage girl teen young woman girl brunette":8, "attractive teenage girl teen young woman girl brunette":8, "baby girl fashion":8, "baby girl sitting on a chair":8, "baby-girl":8, "babygirl":8, "background hijab girls":8, "ballerina girl":8, "barbie girl":8, "beautiful baby girl":8, "beautiful bay girl":8, "beautiful blonde girl":8, "beautiful eyes girl":8, "beautiful girl ashamed":8, "beautiful girl at sunset":8, "beautiful girl eating food":8, "beautiful girl eating takeaway":8, "beautiful girl model":8, "beautiful girl portrait":8, "beautiful girl smiling":8, "beautiful girl wearing engagement ring":8, "beautiful girl wearing glasses":8, "beautiful girl with long hair":8, "beautiful little blonde girl":8, "beautiful little girl":8, "beautiful schoolgirl":8, "beauty-girl":8, "biker girl":8, "bikini girl":8, "biracial girl":8, "birthday girl":8, "black girl":8, "black girls":8, "blogger girl":8, "blond girl":8, "blonde girl natural make up":8, "blossom girl":8, "blue eye girl":8, "blue eyes girl":8, "bond girl":8, "brave little girl":8, "brazilian girl":8, "british girl":8, "brown eyed girl":8, "brown girl":8, "brown hair girl":8, "brunette girl":8, "burlesque girl":8, "business girl":8, "call centre girl":8, "cannabis girl":8, "cannabis girls":8, "cannabis hijab girls":8, "cardigan girl":8, "care free girl":8, "carefree girl":8, "cat-girl":8, "caucasian girl":8, "cbd girl":8, "cbd hijab girls":8, "charming girl":8, "cheeky girl":8, "cheerful girl":8, "child girl":8, "children girl":8, "chinese baby girl":8, "chinese girl":8, "city girl":8, "clever girl":8, "closed eyes girl":8, "clown girl":8, "college girl":8, "confident girl":8, "cool girl":8, "cool girls":8, "cool looking girl":8, "copy spaceyoung girl child children portrait play playing concentrating expression concentration":8, "country girl":8, "cover girl":8, "cow girl":8, "cow girls":8, "cowgirl boots":8, "cowgirl hat":8, "cowgirl silhouette":8, "crazy girl":8, "crying girl":8, "curly haired girl":8, "curvy girl":8, "cute adorable baby girl":8, "cute boy and girl":8, "cute girl face":8, "cute girl laughing":8, "cute girl running":8, "cute kid girl":8, "cute little girl":8, "cute little girl eating":8, "cute little girl in a dress":8, "cute young girl":8, "cutegirl":8, "cyberpunk girl":8, "daddy with a girl":8, "dancing girl":8, "danger girl":8, "danish girl":8, "dark hair girl":8, "dark haired girl":8, "daughter; schoolgirl":8, "delivery girl":8, "determined girl":8, "devil girl":8, "disco girl":8, "dog and girl":8, "dream girl":8, "dreams of a girl":8, "dreamy girl":8, "east asian girl":8, "eastern european girl":8, "edgy girl":8, "elf girl":8, "emo girl":8, "english girl":8, "evil girl":8, "facial hair natural girl":8, "farm-girl":8, "farmer girl":8, "farmgirl":8, "fashion baby girl":8, "fashion girlashion":8, "fashionable girl":8, "female girl":8, "female-girl":8, "fighter girl":8, "finance girls":8, "fit girl":8, "fitness girl":8, "flapper girl":8, "flower crown girl":8, "flower-girl":8, "flowergirl":8, "flying girl":8, "four year old girl":8, "freckled girl":8, "frightened girl":8, "frontiers girl":8, "fun girl":8, "funky girl":8, "funny girl":8, "funny girl eating":8, "funny little girl eating":8, "gas mask girl":8, "geisha girl":8, "generation girlfriend":8, "gesture girl":8, "get girl":8, "ginger girl":8, "gipsy girl":8, "girl adventure":8, "girl age 1":8, "girl and dog":8, "girl and emotions":8, "girl and facial":8, "girl and horse":8, "girl and ice cream":8, "girl and kitten":8, "girl and plants":8, "girl apparel":8, "girl asleep":8, "girl at beach":8, "girl at sunset":8, "girl back view":8, "girl backpack":8, "girl balloons":8, "girl bathing":8, "girl bed":8, "girl bedroom":8, "girl blowing a dandelion":8, "girl blowing bubbles":8, "girl bully":8, "girl carry bag":8, "girl carrying bucket":8, "girl caucasian ethnicity":8, "girl celebrating":8, "girl choosing ice cream":8, "girl climbing":8, "girl computer":8, "girl cooking":8, "girl dog and cat":8, "girl drawing":8, "girl dressed up":8, "girl drinking coffee":8, "girl drinking tea":8, "girl eating apple":8, "girl eating food":8, "girl eating grapes":8, "girl eating healthy":8, "girl eating healthy food":8, "girl eating healthy takeaway food":8, "girl exercise":8, "girl exercising":8, "girl eyes closed":8, "girl face":8, "girl fashion":8, "girl fashionista":8, "girl fever":8, "girl fishing":8, "girl football":8, "girl footballer":8, "girl friend":8, "girl friends":8, "girl gamer":8, "girl garden":8, "girl gardening":8, "girl gym":8, "girl hand":8, "girl happy":8, "girl having fun":8, "girl hemp":8, "girl hemp oil":8, "girl hiking":8, "girl holding a cat":8, "girl holding card":8, "girl holding sandwich":8, "girl hoodie":8, "girl in a dress":8, "girl in a field":8, "girl in a red coat":8, "girl in autumn park":8, "girl in dress":8, "girl in field of flowers":8, "girl in forest":8, "girl in garden":8, "girl in hat":8, "girl in jeans":8, "girl in kitchen":8, "girl in nature":8, "girl in pink":8, "girl in red":8, "girl in red coat":8, "girl in shirt":8, "girl in stables":8, "girl in summer hat":8, "girl in the snow":8, "girl in tight jeans":8, "girl in twenties":8, "girl jeans":8, "girl jogging":8, "girl jump":8, "girl jumping":8, "girl jumping in the sea":8, "girl kissing":8, "girl kissing penguin":8, "girl lady":8, "girl laughing":8, "girl leaning":8, "girl like a bird":8, "girl love art":8, "girl lying down":8, "girl master plumber":8, "girl meditates":8, "girl meditating":8, "girl missing friends":8, "girl model":8, "girl mountaineering":8, "girl mouth":8, "girl next door":8, "girl on a bike":8, "girl on a bmx":8, "girl on a train":8, "girl on beach":8, "girl on cellphone":8, "girl on lavender field":8, "girl on mobile":8, "girl on mobile phone":8, "girl on phone":8, "girl on rocks":8, "girl only":8, "girl outdoors":8, "girl park":8, "girl piercing":8, "girl playing":8, "girl playing football":8, "girl playing in the snow":8, "girl playing soccer":8, "girl playing tennis":8, "girl positive":8, "girl praying":8, "girl reading":8, "girl reading in bed":8, "girl receive":8, "girl riding bicycle":8, "girl riding bike":8, "girl riding motorcycle":8, "girl rugby":8, "girl rugby team":8, "girl runner":8, "girl running":8, "girl running isolated":8, "girl sat down":8, "girl sat in garden":8, "girl silhouette":8, "girl sitting":8, "girl sitting alone":8, "girl sitting on a bench":8, "girl sitting on a wall":8, "girl sleeping":8, "girl smiling":8, "girl sniffing daffodils":8, "girl sniffing flowers":8, "girl soccer":8, "girl soccer player":8, "girl squad":8, "girl taking selfie":8, "girl toddler":8, "girl traveling":8, "girl trekking":8, "girl using phone":8, "girl walking":8, "girl walking at sunset":8, "girl walking dog":8, "girl walking her dog":8, "girl walking up stairs":8, "girl walking up steps":8, "girl wearing glasses":8, "girl wearing pink":8, "girl wearing ppe":8, "girl winking":8, "girl winter":8, "girl with blue hair":8, "girl with chalk":8, "girl with ducks":8, "girl with flowers":8, "girl with gaps in teeth":8, "girl with gun":8, "girl with horse":8, "girl with penguin":8, "girl with piercings":8, "girl with plant":8, "girl-friend":8, "girl.":8, "girl. background":8, "girl. girls":8, "girl. girls":8, "girl's":8, "girl's brunch":8, "girl's neck":8, "girlchinese":8, "girlfriend hand":8, "girlfrined":8, "girlhood":8, "girlie":8, "girlish":8, "girlpower":8, "girls basketball":8, "girls eyes":8, "girls fashion":8, "girls night club":8, "girls nightclub":8, "girls on the park":8, "girls power":8, "girls room":8, "girls rule":8, "girls talk":8, "girls time":8, "girls trip":8, "girls with fancy dress":8, "girls' football":8, "girls' night":8, "girlsmall":8, "girlsnurturing":8, "girlteddywindow":8, "glamour girl":8, "good girl":8, "gorgeous baby girl":8, "gorgeous girl":8, "goth girl":8, "gothic girl":8, "greek girl":8, "green eye girl":8, "grinning girl":8, "grumpy girl":8, "grunge girl":8, "gun-girl":8, "gym girl":8, "gymnastic girl":8, "happy baby girl":8, "happy girl":8, "happy girl face":8, "happy girlfriend":8, "happy girls":8, "happy little girl":8, "happy teenage girl":8, "health girl":8, "healthy girl":8, "heartbroken girl":8, "hemp girl":8, "hemp hijab girls":8, "high school girl":8, "hijab girl":8, "hijab girls":8, "hipster girl":8, "home girl":8, "horror girl":8, "indian girl":8, "indian girl child":8, "indian girl with long hair":8, "indonesia girl":8, "indonesian girl":8, "indonesian girls":8, "innocent girl":8, "innocent young girl":8, "island girl":8, "japanese baby girl":8, "jeans girl":8, "joyful girl":8, "kind girl":8, "korean girl":8, "laptop girl":8, "latin girl":8, "latina girl":8, "laughing girl":8, "lithuanian girl":8, "litte girl":8, "littl girl":8, "little blonde girl":8, "little girl alone":8, "little girl and her dog":8, "little girl asleep":8, "little girl asleep outdoors":8, "little girl at a rainy window":8, "little girl at ickworth":8, "little girl buried in sand":8, "little girl digging":8, "little girl eating a sausage roll":8, "little girl eating italian food":8, "little girl having fun":8, "little girl hiking":8, "little girl in a red coat":8, "little girl in a white dress":8, "little girl in garden":8, "little girl in the forest":8, "little girl in the snow":8, "little girl in the woods":8, "little girl jaywick":8, "little girl laughing":8, "little girl lost":8, "little girl on beach":8, "little girl paddling":8, "little girl painting":8, "little girl playing":8, "little girl playing in the sea":8, "little girl plying":8, "little girl running":8, "little girl sandcastles":8, "little girl skateboarding":8, "little girl smiling":8, "little girl studying":8, "little girl sunset":8, "little girl tennis":8, "little girl walking":8, "little girl walking by the beach":8, "little girl walking in jaywick":8, "little girl walking in the snow":8, "little girl walking to school in the snow":8, "little girl walking towards a spooky house":8, "little girl wearing armbands in swimming pool":8, "little girl with flower":8, "little happy girl":8, "london girl":8, "lonely girl":8, "long hair girl":8, "lovely girl":8, "malaysia girls":8, "malaysian girl":8, "malaysian girls":8, "manga girl":8, "masked girl":8, "messy hair girl":8, "military girl":8, "mixed race girl":8, "modern girl":8, "moroccan girl":8, "naked girl":8, "naked goth girl":8, "natural girl":8, "natural girl face":8, "neon girl":8, "nude girl":8, "nude goth girl":8, "office girl":8, "one girl":8, "one little girl":8, "one little girl only":8, "one teenage girl":8, "only one baby girl":8, "only one girl":8, "only one pre-adolescent girl":8, "only one teenage girl":8, "open mouth girl":8, "oriental school girl":8, "owgirl":8, "pakistani girl":8, "pale faced girl":8, "party girl":8, "pensive girl":8, "person girl":8, "pin up girl":8, "pin up girl vintage":8, "pink hair girl":8, "pinup girl":8, "playgirl":8, "policegirl":8, "polish girl":8, "polish girl in uk":8, "positive girl":8, "poster girl":8, "power girl":8, "pre teen girl":8, "pre teen girls":8, "pre-adolescent girl":8, "pre-teen girl":8, "pre-teen girls":8, "premature baby girl":8, "preteen girl":8, "preteen girls":8, "pretty asian girl":8, "pretty girl face":8, "pretty girl growing teeth":8, "pretty girl on a boat":8, "pretty girl smiling":8, "pretty girl wearing a leather flying helmet and goggles":8, "pretty little girl":8, "pretty young girl in the snow":8, "princess girl":8, "proud girl":8, "rebel girl":8, "red dress girl":8, "red hair girl":8, "red haired little girl":8, "red head girl":8, "red-haired girl":8, "redhead girl":8, "redhead model girl":8, "relaxed beautiful brunette girl":8, "retro image of baby girl":8, "rock girl look":8, "rocker girl":8, "rural girl":8, "russian girl":8, "sales girl":8, "salesgirl":8, "santa girl":8, "scary girl":8, "school girl":8, "school girl costume":8, "school girl uniform":8, "school girls":8, "school-girl":8, "schoolergirl":8, "schoolgirl costume":8, "schoolgirls":8, "sci fi girl":8, "seductive girl":8, "selfie girl":8, "sensual girl":8, "sexy girls":8, "sexy santa girl":8, "shorts girl":8, "show girl":8, "shy girl":8, "silhouette girl":8, "single girl":8, "sitting girl":8, "sitting girl with violin":8, "skateboarding little girl":8, "skater girl":8, "skatergirl":8, "skin care girl":8, "sleeping girl":8, "slim girl":8, "slim girl measuring":8, "smal girl":8, "small girl":8, "small girl reading":8, "small girls":8, "smart girl":8, "smiley girl":8, "smiling girl":8, "smiling girls":8, "smiling little girl":8, "solo girl":8, "space girl":8, "special girl":8, "sport girl":8, "sport-girl":8, "sportsgirl":8, "sporty girl":8, "stable girl":8, "stablegirl":8, "steampunk girl":8, "strong girl":8, "student girl":8, "student girl portrait":8, "studio shot of cute young girls":8, "stylish girl":8, "super girl":8, "surfer girl":8, "surfgirl":8, "surfing girl":8, "surprised girl":8, "swedish girl":8, "sweet girl":8, "sweet little girl":8, "swimsuit girl":8, "tanned girl":8, "tattoo girl":8, "teen girl":8, "teenage girl face":8, "teenage girl portrait":8, "teenaged girl":8, "teenaged girls":8, "teenager girl":8, "tennis girl":8, "thailand girls":8, "thinking girl":8, "thoughtful girl":8, "three year old girl":8, "tied hair young girl":8, "tiny girl":8, "toddle girl":8, "toddler girl":8, "toddler girl dress":8, "toddler girl playing with magnet board":8, "toddler girl playing with play doh":8, "toddler girls":8, "tomboy girl":8, "travel girl":8, "travel girls":8, "trendy girl":8, "two young girls playing outdoors":8, "unhappy girl":8, "upset girl":8, "victorian girl":8, "walking girl":8, "wellness girl":8, "white dress girl":8, "woman girl":8, "working girl":8, "worried girl":8, "young girl at the hairdressers":8, "young girl celebrates in studio":8, "young girl climbing":8, "young girl fashion":8, "young girl hiking":8, "young girl in the lake":8, "young girl in woods":8, "young girl looking out to the city":8, "young girl lying down":8, "young girl outdoors":8, "young girl playing":8, "young girl playing outdoors":8, "young girl portrait":8, "young girl running":8, "young girl smiling":8, "young girl wearing mask":8, "young girl.":8, "young teenage girl":8, "zombie girl":8, "1800s woman":8, "1900s woman":8, "1920s woman":8, "1930s woman":8, "1940s woman":8, "1950s woman":8, "1960s woman":8, "20s woman":8, "30s woman":8, "30s woman portrait":8, "40 something woman":8, "40s woman":8, "50s woman":8, "55 year old woman":8, "a woman":8, "a woman alone":8, "a young white woman":8, "abuse woman":8, "abused woman":8, "acne woman":8, "active woman":8, "active woman outside":8, "adult woman":8, "adult woman study word motivation inspiration graphic lgiht bulb":8, "adult womanlt":8, "afraid woman":8, "african american woman":8, "african american woman smiling":8, "african american woman swimming":8, "african woman":8, "african woman isolated":8, "afro american woman":8, "afro woman":8, "aged woman":8, "aging woman":8, "airwoman":8, "alluring woman":8, "alone woman":8, "amazed woman":8, "angry woman":8, "annoyed woman":8, "anxious woman":8, "arrested woman":8, "arrogant woman":8, "asian business woman":8, "asian businesswoman":8, "asian businesswoman with documents":8, "asian businesswoman with globe":8, "asian businesswoman with mobile":8, "asian businesswoman with santa hat":8, "asian businesswoman with tablet":8, "asian businesswoman writing":8, "asian woman":8, "asian woman doctor":8, "asian woman face":8, "asian woman smiling":8, "asian young woman":8, "athletic woman":8, "attitude woman":8, "attraction woman":8, "attractive business woman":8, "attractive older woman":8, "attractive teenage girl teen young woman girl brunette":8, "attractive woman body":8, "attractive woman face":8, "attractive woman in red lingerie":8, "attractive woman on holidays":8, "attractive woman smiling":8, "attractive young woman":8, "back view woman":8, "bald woman":8, "barwoman":8, "beach woman":8, "beaten woman":8, "beautiful asian woman":8, "beautiful attractive woman naked":8, "beautiful blonde woman":8, "beautiful eyes woman":8, "beautiful face woman":8, "beautiful older woman":8, "beautiful senior woman":8, "beautiful smiling woman":8, "beautiful woman body":8, "beautiful woman by the sea":8, "beautiful woman eating":8, "beautiful woman enjoying nature":8, "beautiful woman face":8, "beautiful woman in woodland":8, "beautiful woman outdoors":8, "beautiful woman portrait":8, "beautiful woman sat on log":8, "beautiful woman wearing a scarf":8, "beautiful woman wearing glasses":8, "beautiful woman wearing warm clothes":8, "beauty woman":8, "beauty-woman":8, "beutiful woman":8, "bikini woman":8, "biracial woman":8, "birthday woman":8, "black beautiful woman":8, "black dress woman":8, "black hair woman":8, "black professional woman":8, "black woman":8, "black woman beauty":8, "black woman face":8, "black woman smiling":8, "blazer woman":8, "blindfold woman":8, "blond business woman":8, "blond haired woman":8, "blond woman":8, "blond woman mask":8, "blond woman smiling":8, "blonde business woman":8, "blonde businesswoman":8, "blonde hair woman":8, "blonde woman dress":8, "blonde woman red coat":8, "blonde woman smiling":8, "blonde woman white background":8, "blonde woman with a camera":8, "blue dress woman":8, "blue eyes woman":8, "blue hair woman":8, "blurred woman":8, "body shape woman":8, "body woman":8, "boots woman":8, "bored woman":8, "boss woman":8, "brave woman":8, "british woman":8, "brown eyes woman":8, "brown hair woman":8, "brown haired woman":8, "brunette business woman":8, "brunette woman":8, "buinesswoman":8, "buisness-woman":8, "burlesque woman":8, "business business woman, student":8, "business businesswoman":8, "business woman isolated":8, "business woman, beautiful":8, "business-woman":8, "business, business woman":8, "business, business woman, student":8, "business. business woman":8, "businesswoman call":8, "businesswoman calm":8, "businesswoman candid":8, "businesswoman chat":8, "businesswoman with mobile":8, "businesswoman with tablet":8, "busineswoman":8, "bussinesswoman":8, "busy woman":8, "buttocks woman":8, "call center woman":8, "calm woman":8, "camerawoman":8, "candid woman":8, "career woman":8, "cartwheeling woman":8, "casual woman":8, "cat and woman":8, "cat woman":8, "catwoman":8, "caucasian woman":8, "caucasian woman beach":8, "caucasian woman on sandy beach":8, "caucasian woman smiling":8, "caucasian-woman":8, "cave woman":8, "celebration woman":8, "ceo woman":8, "charming woman":8, "charming young woman":8, "charwoman":8, "cheeky woman":8, "cheerful woman":8, "chinese woman":8, "chubby woman":8, "churchwoman":8, "classy woman":8, "clever woman":8, "close eye woman":8, "close up portrait woman":8, "close up woman":8, "closed eyes woman":8, "cold woman":8, "concerned woman":8, "confidence woman":8, "confident woman":8, "confused woman":8, "congresswoman":8, "contemplating woman":8, "contemplative woman":8, "contented woman":8, "coronavirus elderly woman isolation":8, "coughing woman":8, "crazy woman":8, "creative woman finding inspiration online":8, "crouching woman":8, "crying woman":8, "curly hair woman":8, "curly woman":8, "curvy woman":8, "customer service woman":8, "cute asian woman":8, "cute woman.":8, "dancer woman":8, "dark haired woman":8, "dark skin woman":8, "daydream woman":8, "daydreaming woman":8, "delighted woman":8, "depressed woman":8, "determination woman":8, "determined woman":8, "diet woman":8, "disappointed woman":8, "disheveled woman":8, "dog and woman":8, "dreaming woman":8, "dreamy woman":8, "drinking wine woman":8, "easter woman":8, "eastern european woman":8, "elder woman":8, "elderly woman isolated":8, "elderly woman on phone":8, "elegance woman":8, "elegant dress woman":8, "elegant woman":8, "embarrassed woman":8, "emo woman":8, "emotional one woman":8, "emotional woman":8, "emotionless woman":8, "empowered woman":8, "energetic woman":8, "england woman":8, "english woman":8, "envy woman":8, "ethnic woman portrait":8, "everywoman":8, "evil woman":8, "excited woman":8, "executive woman":8, "executive woman isolated":8, "exercise woman":8, "extravagant woman":8, "eye-shadow woman":8, "eyes closed woman":8, "face woman":8, "facewoman":8, "family woman":8, "fantasy woman":8, "fashion model woman":8, "fashion portrait woman":8, "fashion woman":8, "fashion woman portrait":8, "fashionable woman":8, "fatigue woman":8, "fed up woman":8, "female woman":8, "female woman 40s forties comforting profession occupation career":8, "fetish woman":8, "fierce woman":8, "fifties woman":8, "filipino woman":8, "firewoman":8, "fisher woman":8, "fit body woman":8, "fit woman":8, "fitness woman":8, "flat woman":8, "flexible woman":8, "flirting woman":8, "floating woman":8, "focused woman":8, "forewoman":8, "forties woman":8, "freckled woman":8, "freckles woman":8, "free woman":8, "freedom woman":8, "fresh face woman":8, "friendly woman":8, "frightened woman":8, "frustrated woman":8, "full body woman":8, "full body woman blonde":8, "full length woman":8, "funny face woman":8, "funny woman":8, "future woman":8, "gardener woman":8, "gardening woman":8, "glamorous woman":8, "glamour woman":8, "glaring woman":8, "glasses woman":8, "gold woman":8, "golden hair woman":8, "good looking woman":8, "gorgeous hispanic woman":8, "gorgeous woman":8, "gorgeous woman fashion smiling cheerful":8, "gothic woman":8, "graceful woman":8, "greek woman":8, "green eyes woman":8, "grey hair woman":8, "grey haired woman":8, "grieving woman":8, "hair care woman":8, "hair dryer woman":8, "hair model woman":8, "haircut woman":8, "hairstyle woman":8, "handcuffed woman":8, "handkerchief woman":8, "handy woman":8, "happiness woman":8, "happiness woman smiling casual studio portrait":8, "happy smiling woman":8, "happy woman smile":8, "happy womann":8, "happy young woman":8, "hapy woman":8, "hat woman":8, "head shot woman":8, "headache woman":8, "headphones woman":8, "headset woman":8, "headshot woman":8, "healthy eating woman":8, "healthy woman":8, "healthy woman happy":8, "heartbroken woman":8, "hijab healths woman":8, "hijab woman":8, "hike woman":8, "hiking woman":8, "hispanic woman":8, "holiday maker woman":8, "hooded woman":8, "hopeful woman":8, "horrified woman":8, "hot gothic woman":8, "hot woman":8, "hot woman beach":8, "house woman":8, "independent woman":8, "independent woman senior":8, "indian woman":8, "indian woman doctor":8, "indian woman driving":8, "indonesian woman":8, "innocent woman":8, "intelligent woman":8, "international woman":8, "international womans day":8, "isolated woman":8, "isolated woman standing":8, "italian woman":8, "japanese woman":8, "jeans woman":8, "joyful woman":8, "jumping woman":8, "kinky woman":8, "kneeling woman":8, "latin woman":8, "latino woman":8, "laughing woman":8, "layers of woman":8, "leather jacket woman":8, "leggings woman":8, "legs woman":8, "lifestyle woman":8, "lingerie woman":8, "lithuanian woman":8, "little old woman":8, "little woman":8, "lone woman":8, "lonely senior woman":8, "lonely woman":8, "long hair woman":8, "long haired woman":8, "looking away woman happy":8, "looking woman":8, "lovely woman":8, "loving woman":8, "low key woman":8, "make up woman":8, "married woman":8, "mature sexy woman":8, "medical woman":8, "meditation woman":8, "mid adult woman":8, "mid aged woman":8, "mid-adult woman":8, "middle age woman":8, "middle aged woman":8, "middle aged woman isolated":8, "middle aged woman portrait":8, "middle eastern woman":8, "middle-aged woman":8, "military woman":8, "mindful woman":8, "mirror woman":8, "mischievous woman":8, "miserable woman":8, "mixed woman":8, "model hair woman":8, "model woman":8, "model woman blond":8, "model woman summer":8, "modern woman":8, "muscles woman":8, "muscular woman":8, "mysterious woman":8, "naked woman":8, "natural beauty woman":8, "natural woman":8, "naturalwoman":8, "needlewoman":8, "nepalese woman":8, "nervous woman":8, "nice woman":8, "no make up woman":8, "noir woman":8, "normal woman":8, "nude woman":8, "nude woman breast":8, "older woman":8, "older woman face":8, "older woman in a park":8, "older woman portrait":8, "older woman smiling":8, "one mid adult woman":8, "one mid-adult woman only":8, "one middle aged woman only":8, "one only woman":8, "one senior woman":8, "one woman":8, "one young adult woman only":8, "one young woman":8, "only one mature woman":8, "only one mid adult woman":8, "only one senior woman":8, "only one woman":8, "only one young woman":8, "only woman":8, "only woman only":8, "orbusinesswoman":8, "outdoors woman":8, "peaceful woman":8, "pensive woman":8, "perfect body woman":8, "period woman":8, "photographer woman":8, "piercing woman":8, "pink hair woman":8, "playful woman":8, "pleased woman":8, "plus size woman":8, "pointing woman":8, "pondering woman":8, "pool side woman":8, "portrait of a beautiful woman":8, "portrait of beautiful woman":8, "portrait of woman":8, "portrait of young woman":8, "portrait woman smiling":8, "portrit of woman":8, "pose woman":8, "posh woman":8, "possessed woman":8, "postwoman":8, "powerful woman":8, "pregnancy woman":8, "pregnant woman outdoors":8, "pregnant woman reading":8, "pregnant woman relaxing":8, "pretty asian woman":8, "pretty indian woman":8, "pretty sportswoman":8, "pretty woman face":8, "pretty woman lunging":8, "pretty woman smiling":8, "pretty woman stretching":8, "pretty woman yoga":8, "pretty young woman":8, "professional woman":8, "professional woman portrait":8, "professional woman with laptop":8, "profile portrait of a woman":8, "profile woman":8, "proud woman":8, "provocative woman":8, "queer woman":8, "questioning woman":8, "radiant woman":8, "real woman":8, "rear view woman":8, "red carpet woman":8, "red hair woman":8, "red hair woman portrait":8, "red hair woman thinking":8, "red haired woman":8, "red head woman":8, "red headed woman":8, "red lips woman":8, "red shoes woman":8, "redhead woman":8, "reflecting woman":8, "reflection of woman in puddle":8, "relaxed woman":8, "relaxed womanx":8, "relaxing woman":8, "repair woman":8, "repairwoman":8, "resting woman":8, "retired woman":8, "retro style woman":8, "retro woman":8, "rock climbing woman":8, "romantic woman":8, "running woman":8, "sad woman":8, "sad woman face":8, "sad woman on phone":8, "sadness woman":8, "sales woman":8, "scandinavian woman":8, "scared woman":8, "scarf woman":8, "sci fi woman":8, "science woman":8, "screaming woman":8, "seated woman":8, "seduce woman":8, "seductive woman":8, "self employed woman":8, "selfie woman":8, "senior adult woman":8, "senior businesswoman":8, "senior woman drink tea leisure":8, "senior woman only":8, "seniorwoman":8, "sensual woman":8, "sensual woman portrait":8, "sensual woman profile":8, "sequencewoman":8, "serene woman":8, "serious woman":8, "servicewoman":8, "sexual woman":8, "sexy beautiful woman":8, "sexy blonde woman laid on side on a black glossy table":8, "sexy woman lingerie":8, "sexy young woman":8, "shocked woman":8, "shocked woman face":8, "short hair woman":8, "shouting woman":8, "shy woman":8, "side profile woman":8, "side view of woman":8, "silhouette woman":8, "single woman":8, "sitting woman":8, "sleeping woman":8, "sleeping woman bed":8, "sleeping woman morning":8, "sleepy woman":8, "slender woman":8, "slender woman beach":8, "slim body woman":8, "slim woman":8, "slim woman body":8, "smart looking woman":8, "smart woman":8, "smile woman":8, "smilewoman":8, "smiling tourist woman":8, "smiling woman face":8, "smiling woman white background":8, "smoking woman":8, "smooth skin woman":8, "solitary woman":8, "south asian woman":8, "southeast asian woman":8, "spa woman":8, "sport woman":8, "sport woman isolated":8, "sports woman":8, "sporty woman":8, "springtime woman":8, "stalking woman":8, "standing woman":8, "standing woman isolated":8, "stern woman":8, "stockings woman":8, "street style woman":8, "stressed out woman":8, "stressed woman":8, "stretching woman":8, "stripper woman":8, "strong independent woman":8, "student woman":8, "studio portrait woman":8, "studio woman":8, "stunning woman":8, "style woman":8, "stylish african woman":8, "stylish woman":8, "stylish woman fashion":8, "stylish young woman":8, "success woman":8, "successful business woman":8, "successful woman":8, "sultry woman":8, "sun glasses woman":8, "sun hat woman":8, "sunbathing woman":8, "sunglasses woman":8, "sunset woman":8, "suntan woman":8, "super woman":8, "suprised woman":8, "surprised woman":8, "suspicious woman":8, "swedish woman":8, "swimsuit woman":8, "swimwear woman":8, "swoman":8, "tall woman":8, "tank top woman":8, "tanned woman":8, "tanned woman beach":8, "tanned woman isolated":8, "tanning woman":8, "tattoo woman":8, "tattooed woman":8, "telephone asian woman":8, "tennis woman":8, "terrified woman":8, "the woman":8, "thin woman":8, "thinking woman":8, "thoughtful woman":8, "thumbs up woman":8, "tights woman":8, "tired woman":8, "toned woman":8, "topless woman":8, "torso woman":8, "tough woman":8, "tourist woman":8, "tradeswoman":8, "tranquility woman":8, "trans woman":8, "travel woman":8, "trendy woman":8, "turban woman":8, "ukrainian woman":8, "underwater woman":8, "underwear woman":8, "undressed woman":8, "unfaithful woman":8, "unhappy woman":8, "uniform woman":8, "upset woman":8, "vegetarian woman":8, "veiled woman":8, "very attractive young woman":8, "victorian woman":8, "victorian woman hat":8, "vintage woman":8, "violence against woman":8, "violence woman":8, "vitality woman":8, "voluptuous woman":8, "vulnerable woman":8, "walking woman":8, "warm woman":8, "warrior woman":8, "weight loss woman":8, "well aged woman":8, "well dressed woman":8, "wellbeing woman":8, "western woman":8, "wet hair woman":8, "wet woman":8, "white hair woman":8, "windy woman":8, "winner woman":8, "winter fashion woman":8, "winter woman":8, "woman 70s":8, "woman ache":8, "woman alone":8, "woman and animal":8, "woman and child":8, "woman and flowers":8, "woman and nature":8, "woman applying make up":8, "woman applying mascara":8, "woman artist":8, "woman at camera":8, "woman at party":8, "woman at peace":8, "woman at sunset":8, "woman at the peak":8, "woman at the top":8, "woman at work":8, "woman attractive":8, "woman back view":8, "woman background":8, "woman backpack":8, "woman balloon":8, "woman balloons":8, "woman beach":8, "woman beach watching":8, "woman being chased":8, "woman body":8, "woman bosses":8, "woman breakfast healthy":8, "woman brown hair":8, "woman by lake":8, "woman by the sea":8, "woman carpenter":8, "woman catching taxi":8, "woman caucasian":8, "woman celebrating":8, "woman chaser":8, "woman chicken salad":8, "woman christmas":8, "woman christmas cake":8, "woman christmas crackers":8, "woman coat":8, "woman cold":8, "woman corset":8, "woman count many currency":8, "woman culture":8, "woman dancing at sunset":8, "woman day":8, "woman decorating":8, "woman deep in thought":8, "woman dentist":8, "woman depressed":8, "woman discomfort":8, "woman diver":8, "woman doctor":8, "woman doing yoga":8, "woman drinking":8, "woman drinking alcohol":8, "woman drinking beer":8, "woman drinking water":8, "woman drinking wine":8, "woman driver":8, "woman eating":8, "woman eating aduki beans":8, "woman eating baked beans":8, "woman eating breakfast":8, "woman eating chocolate bar":8, "woman eating grapefruit":8, "woman eating pizza":8, "woman eating salad":8, "woman eating salmon salad":8, "woman eating sausage":8, "woman eating takeaway food":8, "woman enjoying life":8, "woman entrepreneur":8, "woman exercising":8, "woman exercising outdoors":8, "woman explaining":8, "woman eyes":8, "woman fashion model":8, "woman fashion portrait":8, "woman female":8, "woman frown":8, "woman frowning":8, "woman gardening":8, "woman girl":8, "woman gun":8, "woman hair":8, "woman hairless":8, "woman hand":8, "woman hand holding":8, "woman hands":8, "woman happiness":8, "woman happy":8, "woman having fun":8, "woman hidden":8, "woman hiding behind balloon":8, "woman hiker":8, "woman hiking":8, "woman holding":8, "woman holding blank sign":8, "woman holding cassette":8, "woman holding ipad":8, "woman holding oranges":8, "woman holding shoes":8, "woman holding sign":8, "woman holding tablet":8, "woman holding vhs":8, "woman holding wine":8, "woman holding wine glass":8, "woman holiday":8, "woman holidays":8, "woman home":8, "woman house":8, "woman i bed":8, "woman in a bedroom":8, "woman in a dress":8, "woman in a hat":8, "woman in a mirror":8, "woman in bath":8, "woman in bed":8, "woman in bed in lingerie":8, "woman in bedroom":8, "woman in bikini":8, "woman in black evening dress":8, "woman in bra":8, "woman in business":8, "woman in car":8, "woman in charge":8, "woman in coat":8, "woman in counter":8, "woman in dress":8, "woman in field":8, "woman in flowers":8, "woman in forest":8, "woman in glasses":8, "woman in hat":8, "woman in iran":8, "woman in kitchen":8, "woman in lingerie":8, "woman in mask":8, "woman in nature":8, "woman in necklace":8, "woman in office":8, "woman in stockings":8, "woman in suit":8, "woman in summer hat":8, "woman in sun hat":8, "woman in sunglasses":8, "woman in thought":8, "woman in white lingerie":8, "woman in yoga pose":8, "woman isolated":8, "woman isolated white":8, "woman kneeling":8, "woman laughing":8, "woman learn":8, "woman legs":8, "woman legs isolated":8, "woman life freedom":8, "woman lifting":8, "woman lifting weight":8, "woman lingerie":8, "woman little":8, "woman looking":8, "woman looking at watch":8, "woman lunging":8, "woman measuring":8, "woman meditates":8, "woman meditating":8, "woman meditating outdoors":8, "woman milk":8, "woman mockup":8, "woman model":8, "woman nature":8, "woman offering fruit":8, "woman on beach":8, "woman on beach smiling":8, "woman on bed":8, "woman on bench":8, "woman on bike":8, "woman on cell phone":8, "woman on country road":8, "woman on edge":8, "woman on phone":8, "woman on piano":8, "woman on stairs":8, "woman on the sofa":8, "woman on the swings":8, "woman on toilet":8, "woman on top":8, "woman only":8, "woman outdoors":8, "woman outdoors smiling":8, "woman outside":8, "woman overwork":8, "woman painting":8, "woman paparazzi":8, "woman photo":8, "woman photojournalism":8, "woman photos":8, "woman playing":8, "woman playing golf":8, "woman pointing finger":8, "woman potter":8, "woman rapid flow":8, "woman ravioli":8, "woman reading":8, "woman reading outdoors":8, "woman recovery":8, "woman relaxing":8, "woman relaxing at home":8, "woman resting":8, "woman resting in office":8, "woman rights":8, "woman running":8, "woman sad":8, "woman sat":8, "woman scared":8, "woman scream":8, "woman scuba diver":8, "woman seated":8, "woman secret":8, "woman selfie":8, "woman sexy":8, "woman shades":8, "woman shadow":8, "woman shadows":8, "woman shocked":8, "woman shopping":8, "woman sick":8, "woman sign":8, "woman silhouette":8, "woman silhouette head":8, "woman silhouette isolated":8, "woman sitting":8, "woman sitting on grass":8, "woman sitting on the beach":8, "woman sitting outdoors smiling":8, "woman skincare":8, "woman sleeping":8, "woman sleeping in bed":8, "woman smart phone":8, "woman smiling":8, "woman smoking":8, "woman sneezing":8, "woman soldier":8, "woman spaghetti":8, "woman standing":8, "woman stockings":8, "woman strawberries":8, "woman stretching":8, "woman stripping":8, "woman studying":8, "woman sunbathing":8, "woman talking":8, "woman talking on phone":8, "woman talking on the phone":8, "woman talking with colleague and using laptop":8, "woman thinking":8, "woman tourist":8, "woman traveller":8, "woman twirling at sunset":8, "woman underwear":8, "woman violence":8, "woman waiting":8, "woman walking at sunset":8, "woman walking away":8, "woman walking isolated":8, "woman washing face":8, "woman washing hands":8, "woman watching sea":8, "woman water":8, "woman watering flower":8, "woman wearing a hat":8, "woman wearing face mask":8, "woman wearing glasses":8, "woman wearing mask":8, "woman wearing stockings":8, "woman wearing sunglasses":8, "woman wine glass":8, "woman wink":8, "woman with a camera":8, "woman with a dslr":8, "woman with camera":8, "woman with caps":8, "woman with face mask":8, "woman with glasses":8, "woman with laptop":8, "woman with mask":8, "woman with nails":8, "woman with nature":8, "woman with wine":8, "woman with wine glass":8, "woman wondering":8, "woman work":8, "woman working":8, "woman working in kitchen":8, "woman working in office":8, "woman working on office":8, "woman yes":8, "woman yoga":8, "woman yoga stretching":8, "woman young":8, "woman-eyes":8, "woman-face":8, "woman. female":8, "woman. headphones":8, "woman'":8, "woman's":8, "woman's face":8, "woman's hand":8, "woman's profile":8, "woman's salon":8, "woman's tennis":8, "woman]":8, "woman8":8, "womanbeautiful":8, "womane":8, "womanish":8, "womanishness":8, "womanism":8, "womanize":8, "womanizer":8, "womanlab":8, "womanlike":8, "womanly":8, "womanly young":8, "womanrunning":8, "womans":8, "womans bottom":8, "womans clothing":8, "womans face 30 years":8, "womans fingers":8, "womans hand":8, "womans hands":8, "womans nails":8, "womans rights":8, "womans shoe":8, "womans wear":8, "womanyoga":8, "wondering woman":8, "working woman":8, "workout woman":8, "worried woman":8, "wrist watch woman":8, "wwoman":8, "yawning woman":8, "yoga poses woman":8, "yoga woman":8, "yong woman":8, "you woman":8, "young adult adult woman":8, "young adult woman":8, "young adultwoman":8, "young afro woman":8, "young attractive woman":8, "young attractive woman in red lingerie":8, "young brown hair woman":8, "young business woman isolated":8, "young businesswoman":8, "young hispanic woman nude":8, "young naked woman":8, "young nude woman":8, "young pretty woman":8, "young woman cartwheeling":8, "young woman doing yoga":8, "young woman face":8, "young woman happy":8, "young woman isolated":8, "young woman meditating":8, "young woman modern":8, "young woman portrait":8, "young woman reading":8, "young woman sitting":8, "young woman sitting on floor":8, "young woman smiling":8, "young woman standing":8, "young woman thinking":8, "young woman ubud temple":8, "young woman washing car":8, "young woman white background":8, "young-woman":8, "20s female":8, "20s2 female":8, "30s female":8, "adolescent female":8, "adult female":8, "ai female eye":8, "american female":8, "athletic female":8, "attractive beautiful investing female person":8, "blond hair female model":8, "blonde female":8, "blonde haired female model":8, "dropper female":8, "face female":8, "fashion female":8, "fashionfemale":8, "female adult 20-25 years":8, "female athlete":8, "female beautiful model":8, "female bird watcher":8, "female caucasian":8, "female caucasian color colour":8, "female celebrating":8, "female clothing":8, "female distance":8, "female drinker":8, "female drivers":8, "female education":8, "female eyes":8, "female feeling":8, "female feet":8, "female financial":8, "female focused":8, "female freelance":8, "female freelancer":8, "female friends":8, "female graduate":8, "female guitarist":8, "female hand holding slice sweet potato":8, "female hands holding":8, "female headset":8, "female healer":8, "female hobby":8, "female icon":8, "female in her twenties":8, "female jogger":8, "female legs":8, "female looking":8, "female mechanic":8, "female model boho":8, "female model in pink bikini":8, "female model landscape":8, "female nature":8, "female nature photographer":8, "female office worker":8, "female one person":8, "female owned business":8, "female paparazzo":8, "female pediatrician":8, "female photojournalist":8, "female pilot":8, "female portraits":8, "female runner":8, "female senior":8, "female shutterbug":8, "female silhouettes":8, "female sitting":8, "female skateboarder":8, "female stressed":8, "female strong":8, "female trekker":8, "female twitchers":8, "female violence":8, "female white":8, "female woman":8, "female-girl":8, "female.":8, "female30s":8, "handcuffed female":8, "hot female":8, "lady female":8, "large busted asian female model":8, "large female chest":8, "large female chested model":8, "lipstick female":8, "lone female":8, "lying female":8, "one female":8, "one female adult only":8, "park female":8, "professional female":8, "professional female photographer":8, "red headed female":8, "redheaded female":8, "sad female":8, "sexy blonde haired female model in red lingerie":8, "sexy large chested female model in red bra":8, "single female traveller":8, "sporty female":8, "terrified female":8, "two adult females":8, "wireless female":8, "17 year old boy":1, "3 years old boy":1, "4 year old boy":1, "5 year old boy":1, "7 year old boy":1, "8 year old boy":1, "a boy lies on a green grass":1, "active boy":1, "active little boy":1, "adhd boy swimming":1, "adolescent boy":1, "adorable boy":1, "again boy":1, "angry boy":1, "asian boy":1, "asian boy face":1, "autistic boy":1, "autistic boy in snow":1, "b-boy":1, "b-boying":1, "baby boy in yellow":1, "baby-boy":1, "babyboy":1, "badboy":1, "ball boy":1, "bayboy":1, "bboy":1, "beautiful boy":1, "behavior boy":1, "bell boy":1, "bellboy":1, "birthday boy":1, "black boy":1, "blond boy":1, "blonde boy":1, "blue eyed boy":1, "boy aged 5":1, "boy alone":1, "boy and dog":1, "boy and milk drink":1, "boy body":1, "boy brushing":1, "boy brushing his teeth":1, "boy brushing teeth":1, "boy catching fish":1, "boy celebration":1, "boy child":1, "boy climbing ladder":1, "boy clothes":1, "boy cutting":1, "boy drinking juice":1, "boy drinks milkshake":1, "boy eating":1, "boy eating apple":1, "boy eating breakfast":1, "boy eating ice cream":1, "boy eating pepper":1, "boy eating snack":1, "boy eating strawberry":1, "boy enjoys milkshake":1, "boy face":1, "boy faces":1, "boy farmer":1, "boy fast asleep":1, "boy fishing":1, "boy football isolated":1, "boy happy":1, "boy having fun":1, "boy hiding behind leaves":1, "boy hiking":1, "boy holding":1, "boy holding a leg":1, "boy in a funny hat":1, "boy in background":1, "boy in blanket":1, "boy in blue":1, "boy in costume":1, "boy in funny hat":1, "boy in halloween costume":1, "boy in life jacket":1, "boy in silhouette":1, "boy in the woods":1, "boy in woods":1, "boy jogging":1, "boy jumping":1, "boy laughing":1, "boy laying down":1, "boy learning":1, "boy looking":1, "boy looking at the view":1, "boy looking down":1, "boy looking out":1, "boy lying":1, "boy model":1, "boy next door":1, "boy on a quad bike":1, "boy on beach":1, "boy on bus":1, "boy on crutches":1, "boy on his belly":1, "boy on his own":1, "boy on left of frame":1, "boy on swing":1, "boy on vacation":1, "boy only":1, "boy outdoors":1, "boy pirate":1, "boy playing":1, "boy playing catch":1, "boy playing football":1, "boy playing in sunset":1, "boy playing outside":1, "boy playing soccer":1, "boy playing with dog":1, "boy playy":1, "boy portrait":1, "boy reading a book":1, "boy reads":1, "boy running":1, "boy running isolated":1, "boy screaming":1, "boy shorts":1, "boy shouting":1, "boy showing tongue":1, "boy silhouette":1, "boy sitting":1, "boy sitting on sidewalk":1, "boy sitting on the floor":1, "boy sleeping":1, "boy smelling":1, "boy smile":1, "boy smiling":1, "boy soccer player":1, "boy teenage":1, "boy thinking":1, "boy toy":1, "boy using a drill":1, "boy valentines day":1, "boy walking":1, "boy walking on a wall":1, "boy walking through water":1, "boy wear sweater":1, "boy wearing tshirt":1, "boy with adhd":1, "boy with adhd in the snow":1, "boy with apple":1, "boy with asperger syndrome":1, "boy with aspergers":1, "boy with aspergers syndrome":1, "boy with autism":1, "boy with autism in swimming pool":1, "boy with dog":1, "boy with freckles":1, "boy with juice":1, "boy with long hair":1, "boy with yellow flowers":1, "boy woodland":1, "boy woods":1, "boy wrapping gift":1, "boy-friend":1, "boy's":1, "boyadorable":1, "boychild":1, "boycotted":1, "boyhood":1, "boys can":1, "boys fashion":1, "boys fun":1, "boys night out":1, "boys playing":1, "boys room":1, "boys toys":1, "boystrous":1, "brown eyed boy":1, "brown hair boy":1, "bullied boy":1, "business boy":1, "caucasian baby boy":1, "caucasian boy":1, "cheeky boy":1, "cheerful boy":1, "child boy":1, "children boy":1, "chinese boy":1, "clever boy":1, "close up boys face":1, "closeup portrait of a boy":1, "cold little boy":1, "cool boy":1, "cow boy":1, "cow boy hat":1, "cropped picture of young boy":1, "crying boy":1, "curly hair boy":1, "cute baby boy":1, "cute boy":1, "cute boy with long hair":1, "cute child boy":1, "cute little boy":1, "cute young boy":1, "cute young boy eating chocolate treat background":1, "dark hair boy":1, "delivery boy":1, "down boy":1, "easter boy":1, "energetic little boy":1, "english boy":1, "enjoymentboy":1, "enthusiastic boy":1, "european boy":1, "farm boy":1, "fashion kid boy":1, "funny boy":1, "funny face boy":1, "gingerbread boy":1, "gree eyed boy":1, "green eyed boy":1, "grinning boy":1, "hair boy":1, "hair style boy":1, "halloween boy":1, "handsome boy smiling":1, "handsome boys":1, "handsome little boy":1, "happy boy":1, "happy boy face":1, "happy boy with adhd":1, "happy boy with aspergers syndrome":1, "happy boys":1, "healthy boy":1, "high school boy":1, "hiking boy":1, "hipster boy":1, "horrible boy":1, "hot boy":1, "infant boy":1, "innocent boy":1, "jean boy":1, "jeans boy":1, "jumping boy":1, "laughing boy":1, "little boy asleep on beach":1, "little boy climbing":1, "little boy climbing a rope":1, "little boy colouring":1, "little boy having fun":1, "little boy in woods":1, "little boy isolated":1, "little boy laughing":1, "little boy on a swing":1, "little boy on beach":1, "little boy playing":1, "little boy portrait":1, "little boy sitting":1, "little boy sitting in a pub":1, "little boy with adhd":1, "little boys":1, "little boys only":1, "little boys woods":1, "littlt boy":1, "lonely boy":1, "looking at boy":1, "mix race boy":1, "mixed race boy":1, "muslim boy":1, "naive boy":1, "naughty boy":1, "newsboy":1, "newsboy cap":1, "one baby boy":1, "one boy":1, "one little boy":1, "one little boy only":1, "one teenage boy":1, "only one baby boy":1, "only one boy":1, "only one pre-adolescent boy":1, "only one teenage boy":1, "page boy":1, "paperboy":1, "pensive boy":1, "photo of a little boy":1, "pizza boy":1, "pool boy":1, "portrait boy":1, "portrait of a little boy":1, "portrait of boy":1, "portrait school boy":1, "pre-adolescent boy":1, "pre-teen boy":1, "preschool boy":1, "pretty boy":1, "red hair boy":1, "red haired boy":1, "red headed boy":1, "school boy":1, "school boy uniform":1, "school uniform boy":1, "school-boy":1, "schoolboy uniform":1, "schoolboys":1, "shirtless boy":1, "shoolboy":1, "silhouette boy":1, "single boy":1, "six year old boy":1, "skaterboy":1, "sleeping baby boy":1, "sleeping boy":1, "small boy":1, "smart boy":1, "smiling boy":1, "smiling face boy":1, "smiling little boy":1, "spartan boy":1, "sporty boy":1, "stable boy":1, "strong boy":1, "student boy portrait":1, "teenage boy portrait":1, "teenage boy smiling":1, "teenaged boy":1, "teenaged boys":1, "teenager boy":1, "teenager boys":1, "thinking boy":1, "thoughtful boy":1, "toddler boy":1, "toddler boy isolated":1, "toddler boys":1, "tom-boy":1, "top boy":1, "unhappy boy":1, "upset boy":1, "viking boy":1, "white boy":1, "young boy angry":1, "young boy eating chocolate":1, "young boy face":1, "young boy isolated":1, "young boy listening to headphones":1, "young boy listening to music":1, "young boy playing":1, "young boy portrait":1, "young boy reading":1, "young boy smiling":1, "18th century man":1, "1940s man":1, "20s man":1, "30 year old man":1, "a man":1, "a yong man watching the sea":1, "a young man":1, "active man":1, "adult man":1, "african american man":1, "african man":1, "aggressive man":1, "alone man":1, "amazed man":1, "angry man":1, "angry white man":1, "annoyed man":1, "anxious man":1, "arabian man":1, "army man":1, "arrogant man":1, "asian man":1, "athletic man":1, "attractive man":1, "attractive older man":1, "attractive young man":1, "axe man":1, "back view man":1, "bald man":1, "balding man":1, "bar man":1, "bare chested man":1, "barefoot man":1, "bathrobe man":1, "beard man":1, "beautiful man":1, "big man":1, "black man":1, "black man face":1, "blindfolded man":1, "blond man":1, "blonde hair man":1, "blonde man":1, "blue eyes man":1, "blue shirt man":1, "bored man":1, "brainy man":1, "brazilian man":1, "british man":1, "broken hearted man":1, "broken man":1, "brown hair man":1, "brown man":1, "brunette man":1, "bulky man":1, "business man isolated":1, "businessman man":1, "calculating man":1, "carpenter man":1, "casual dressed man":1, "casual man":1, "caucasian man":1, "caucasian young man":1, "cheerful man":1, "chinese man":1, "chubby man":1, "city man":1, "classic man fashion":1, "clever man":1, "concerned man":1, "confident man":1, "confident young man":1, "confused man":1, "contemplative man":1, "content man":1, "cool man":1, "cool old man":1, "courier man":1, "craft-man":1, "crazy man":1, "creepy man":1, "crouched man":1, "crouching man":1, "crying man":1, "curious man":1, "dark haired man":1, "dead man":1, "depressed man":1, "depression man":1, "disabled man":1, "diy man":1, "double exposure man":1, "dreaming man":1, "drowning man":1, "elderly happy man":1, "elegant man":1, "elegant man sitting":1, "emotional man":1, "ethnic man":1, "european man":1, "excited man":1, "executive man":1, "expressive man":1, "face man":1, "falling man":1, "family man":1, "fashion man":1, "fashionable man":1, "fat man":1, "fighting man":1, "filipino man":1, "fire man":1, "fit body man":1, "fit man":1, "fitness man":1, "flying man":1, "focused man":1, "frightened man":1, "front-man":1, "frustrated man":1, "full length man":1, "funny man":1, "furious man":1, "gardener man":1, "gazing man":1, "geeky man":1, "ghostly man":1, "ghoul man":1, "ginger hair man":1, "glasses man":1, "gloomy man":1, "goal man":1, "good looking young man":1, "goofy man":1, "gorgeous man":1, "gray hair man":1, "green eyes man":1, "green man":1, "grey hair man":1, "grey haired man":1, "hair removal man":1, "hairy man":1, "handsome looking man":1, "handsome man isolated":1, "handsome man portrait":1, "handsome man smiling":1, "handsome young man":1, "handy man":1, "handy-man":1, "happy man":1, "happy man with phone":1, "happy old man":1, "happy working retired man":1, "hard man":1, "headphones man":1, "healthy living man":1, "healthy old man":1, "high man":1, "hiker man":1, "hipster man":1, "hispanic man":1, "hit man":1, "homeless man":1, "homeless man asleep":1, "homeless man at christmas":1, "homeless man on the street":1, "hooded man":1, "hoodie man":1, "hungover man":1, "indian man":1, "injured man":1, "insomnia man":1, "inspirational man":1, "intelligent man":1, "intense looking man":1, "intimidating man":1, "isolated man":1, "isolated man image":1, "isolated man standing":1, "italian man":1, "japanese man":1, "jazz man":1, "jazz-man":1, "jogging man":1, "joyful man":1, "jumping man":1, "kind face old man":1, "latin man":1, "laughing man":1, "lazy man":1, "leaning man":1, "lens man":1, "lifestyle man":1, "little man":1, "london man":1, "lone man":1, "lone man wearing mask":1, "lonely man":1, "looking man":1, "low key man":1, "macho man":1, "mad man":1, "male man":1, "male. man":1, "man 50 years":1, "man abaya":1, "man age 30":1, "man aged 60":1, "man alone":1, "man and cat":1, "man and dog":1, "man and dog in forest":1, "man and nature":1, "man and phone":1, "man asleep at computer":1, "man asleep at desk":1, "man asleep at his desk":1, "man asleep at work":1, "man asleep on laptop":1, "man asleep on the job":1, "man at home":1, "man at work":1, "man atm":1, "man back view":1, "man background":1, "man backpack":1, "man bag":1, "man betting":1, "man biting a rose":1, "man business":1, "man celebrating":1, "man clapping":1, "man climbing":1, "man dancing":1, "man despair":1, "man drinking from mug":1, "man driving":1, "man eating alone":1, "man eating meal":1, "man enjoying nature":1, "man enjoying nature walk":1, "man exercising":1, "man face":1, "man facial":1, "man feeding the birds over a river":1, "man fishing":1, "man flu":1, "man full of himself":1, "man gambling":1, "man gardening":1, "man glasses":1, "man grass":1, "man grinning":1, "man hand":1, "man hand holding":1, "man hands":1, "man headphones":1, "man hiker":1, "man hiking":1, "man holding a condom":1, "man holding a rose":1, "man holding carp":1, "man holding gun":1, "man holding plate with both hands":1, "man holding playing cards":1, "man housework":1, "man hovering":1, "man in 40's":1, "man in a hat":1, "man in a jumper":1, "man in a suit":1, "man in black":1, "man in cap":1, "man in deep thought":1, "man in forest":1, "man in glasses":1, "man in his 40's":1, "man in jumper":1, "man in leather jacket":1, "man in pain":1, "man in panic":1, "man in rain":1, "man in shadows":1, "man in shock":1, "man in suit":1, "man in the browser":1, "man in truck":1, "man in work clothes":1, "man ironing":1, "man jumping":1, "man keeping quiet":1, "man laughing":1, "man listening to music":1, "man looking at phone":1, "man looking in wonder":1, "man male":1, "man man":1, "man man":1, "man manager":1, "man marry":1, "man medical":1, "man of the woods":1, "man on beach":1, "man on boat":1, "man on mountain":1, "man on phone":1, "man on swing":1, "man on the sofa with remote control":1, "man on the sofa with remote control rejoice":1, "man only":1, "man passing a test":1, "man phone":1, "man phone sea":1, "man phone travel":1, "man photographing forest":1, "man pleasure":1, "man pointing":1, "man portrait":1, "man power":1, "man read map":1, "man reading a book":1, "man red":1, "man remembering":1, "man sat fishing":1, "man sat in parked car":1, "man shaving":1, "man shaving mirror":1, "man silhouette":1, "man sitting":1, "man sitting an":1, "man sitting down":1, "man skipping":1, "man smartphone":1, "man smile":1, "man smiling":1, "man smoking vape":1, "man spreading":1, "man squint":1, "man standing and holding laptop posing for shotoshoot":1, "man standing and posing for photoshoot":1, "man standing bike":1, "man sunbathing":1, "man swimming jungle":1, "man swimming waterfall":1, "man taking photo":1, "man talking":1, "man thinking":1, "man using jump rope":1, "man using laptop":1, "man using skipping rope":1, "man using tools":1, "man vaping":1, "man walk":1, "man walker":1, "man wearing face mask":1, "man wearing mask":1, "man wearing scarf":1, "man wearing suit":1, "man wearing virtual reality headset":1, "man wearing vr":1, "man with a condom":1, "man with a smiling":1, "man with beard":1, "man with camera":1, "man with cap on":1, "man with gun":1, "man with gun isolated":1, "man with horse head":1, "man with idea":1, "man with jump rope":1, "man with laptop":1, "man with mask":1, "man with notepad":1, "man with solution":1, "man with stethoscope":1, "man with stubble":1, "man with target":1, "man with white teeth":1, "man work":1, "man worker":1, "man working":1, "man working late":1, "man working place":1, "man yes":1, "man-bag":1, "man-flu":1, "man-made":1, "man-made model":1, "man.":1, "man's":1, "man's arm":1, "manly man":1, "map man":1, "masculine man":1, "mask man":1, "masked man":1, "mature adult man":1, "mean man":1, "meditation man":1, "mediterranean man":1, "megaphone man":1, "menacing man":1, "metal gate man open":1, "mid adult man":1, "mid-adult man":1, "middle age man":1, "middle aged man":1, "middle aged man smiling":1, "middle aged man standing":1, "middle eastern man":1, "middle-aged man":1, "mischievous man":1, "mix race man":1, "mixed man":1, "mixed race man":1, "model man":1, "modern business man":1, "modern man":1, "muscle man":1, "muscles man":1, "muscular man":1, "muslim man":1, "mustache man":1, "mysterious man":1, "mystery man":1, "naked man":1, "napping man":1, "neck pain man":1, "nerdy man":1, "nervous man":1, "nigerian man":1, "nosy man":1, "odd job man":1, "old man and cat":1, "old man and dog":1, "old man and pets":1, "old man face":1, "old man holding spanner in workshop":1, "old man in red hat working":1, "old man of coniston":1, "old man point":1, "old man portrait":1, "old man singing":1, "old man smiling":1, "old man smiling while working":1, "old man storr":1, "old man working in workshop":1, "old-man":1, "older man":1, "older man portrait":1, "one band man":1, "one man":1, "one man and his dog":1, "one mature man":1, "one mid-adult man only":1, "one senior man":1, "one young adult man":1, "one young man":1, "only man":1, "only one man":1, "only one mature man":1, "only one mid adult man":1, "only one senior man":1, "only one young man":1, "only senior man":1, "optimistic man":1, "outdoors man":1, "overworked man":1, "pakistani man":1, "panicking man":1, "pensive man":1, "piercing man":1, "pilates man":1, "plough man":1, "plus size man":1, "police man":1, "portrait man close up":1, "portrait man smile":1, "professional man":1, "profile man":1, "proud man":1, "quiet man":1, "rabbit man":1, "rasta man":1, "real life old man working":1, "real man":1, "rear view man":1, "red man":1, "regency man":1, "regular man":1, "relationship man":1, "relaxed man":1, "relaxing man":1, "removal man":1, "repair man":1, "resting man":1, "retired man":1, "retired man working":1, "rocker man":1, "romantic man":1, "rough man":1, "rugged man":1, "running man":1, "sad man isolated":1, "sadness man":1, "salary man":1, "salary-man":1, "sales man":1, "scared man":1, "scarry man":1, "scars man":1, "science fiction man":1, "screaming man":1, "secret man":1, "seductive man":1, "senior man portrait":1, "sensual man":1, "serious man":1, "serious man portrait":1, "sexy bearded man":1, "sexy man":1, "shadow man":1, "shadowy man":1, "shaved man":1, "shirtless man":1, "shirtless man sitting":1, "shocked man":1, "short hair man":1, "shouting man":1, "sikh man":1, "silhouette man":1, "silver hair man":1, "singe man":1, "single man":1, "sitting man":1, "sleeping man":1, "slim man":1, "smart casual man":1, "smart man":1, "smartly dressed man":1, "smile man":1, "smoking man":1, "sneezing man":1, "snorkeling man":1, "snow man":1, "solitary man":1, "south asian man":1, "space.man":1, "spanish man":1, "spider man":1, "spider-man":1, "sport man":1, "sport-man":1, "sports man":1, "sporty man":1, "standing man":1, "staring man":1, "stressed man":1, "stretching man":1, "strong man":1, "struggling man":1, "stylish man":1, "succesful man":1, "successful man":1, "suffering man":1, "suit man":1, "sunglasses man":1, "super man":1, "surprised man":1, "suspicious man":1, "sweating man":1, "sweaty man":1, "tall man":1, "tattoo man":1, "tattooed man":1, "tattoos man":1, "tears man":1, "teeth man":1, "the little old man":1, "the man":1, "the old man of storr":1, "thinking man":1, "thoughtful man":1, "threatening man":1, "thumbs up man":1, "tired man":1, "toothless man":1, "torso man":1, "tough man":1, "transgender man":7, "tree man":1, "trendy man":1, "turkish man":1, "unconscious man":1, "underwear man":1, "unhappy man":1, "unknown man":1, "unshaven man":1, "upset man":1, "upside down man":1, "urban man":1, "urban young man":1, "vaping man":1, "violent man":1, "vulnerable man":1, "vuvuzela man":1, "well being elderly man":1, "well dressed business man":1, "well dressed man":1, "white man":1, "white man 3d":1, "white man.":1, "white shirt man":1, "white van man":1, "wide eyed man":1, "wise man":1, "work man":1, "workout man":1, "worried man":1, "wrinkles man":1, "yawning man":1, "yoga man":1, "young adult man":1, "young business man":1, "young handsome man":1, "young man backpack":1, "young man deep in thought":1, "young man in city":1, "young man isolated":1, "young man isolated on white":1, "young man moving":1, "young man portrait":1, "young man posing":1, "young man sitting":1, "young man sitting on floor":1, "young man smiling":1, "young man standing":1, "young man walking":1, "young man with backpack and cap":1, "young man with cap":1, "young man working with laptop":1, "young young man":1, "young-adult man":1, "younger man":1, "15 year old male":1, "adult male":1, "angry male":1, "asian male":1, "attractive male":1, "baby male":1, "bald male":1, "bearded male":1, "blonde male":1, "british male":1, "businessman male":1, "caucasian male":1, "cityscape male":1, "close up male":1, "crying male":1, "dark haired male":1, "dark haired male model":1, "elderly male":1, "eyes male":1, "eyewear male":1, "handsome male":1, "happy male":1, "hat male":1, "healthy male":1, "italian male":1, "life male":1, "lone male":1, "looking away male":1, "male 20s":1, "male 70s":1, "male adult":1, "male adults":1, "male attraction":1, "male beauty":1, "male blogger":1, "male body":1, "male brief":1, "male caucasian":1, "male cheerleader":1, "male child":1, "male close up":1, "male cook":1, "male cool":1, "male doctor":1, "male dress":1, "male face":1, "male facial":1, "male fashion":1, "male female":1, "male fishing":1, "male fitness":1, "male grooming":1, "male hand":1, "male head":1, "male hiker":1, "male man":1, "male mental health":1, "male models":1, "male offspring":1, "male old":1, "male one person":1, "male only":1, "male parenting":1, "male patient":1, "male pattern baldness":1, "male person":1, "male photographer":1, "male portrait":1, "male portrait smile":1, "male runner":1, "male sat fishing":1, "male senior":1, "male senior adult":1, "male senior adults":1, "male shirt":1, "male sitting":1, "male skincare":1, "male skipping":1, "male soccer player":1, "male speaker":1, "male standing":1, "male stare":1, "male student":1, "male sunbathing":1, "male t-shirt":1, "male tank top":1, "male treatment":1, "male using a jump rope":1, "male walker":1, "male walking":1, "male with jump rope":1, "male with skipping rope":1, "male worker eating":1, "male young adult":1, "male. man":1, "man male":1, "masculine male":1, "mature male":1, "men male":1, "mid-adult male":1, "middle aged male":1, "mixed race male":1, "old caucasian male":1, "old male":1, "older male":1, "one male":1, "one male only":1, "one mid adult male only":1, "one young male only":1, "only male":1, "park male":1, "plus size male":1, "prone male":1, "red head male":1, "sad male":1, "senior male":1, "sexy male model":1, "shocked male":1, "smiling male":1, "solo male hiker":1, "solo male walker":1, "toddler male":1, "unhappy male":1, "upset male":1, "white male":1, "white male baby":1, "young adult male":1, "young male":1, "young male in studio":1, "young white male model":1}
gender_dict_both = {"both":11, "men and women":11, "man and woman":11, "male and female":11, "boys and girls":11, "boy and girl":11, }

# gender2key = {"man":"men", "woman":"women"}
eth_dict = {"black":1, "african-american":1, "afro-american":1, "africanamerican":1, "african american":1, "african":1, "indigenous peoples of africa":1, "african ethnicity":1, "african-american ethnicity":1, "african descent":1, "african descen":1, "caucasian":2, "caucasian ethnicity":2, "white people":2, "europeans":2, "eastasian":3,"east asian":3, "chinese":3, "japanese":3, "asian":3, "hispaniclatino":4, "latino":4, "latina":4, "latinx":4, "hispanic":4, "mexican":4, "middleeastern":5, "middle eastern":5, "arab":5, "mixedraceperson":6, "mixedrace":6, "mixed-race":6, "mixed race":6, "mixed ethnicity":6, "multiethnic":6, "multi ethnic":6, "multi-ethnic":6, "biracial":6, "nativeamericanfirstnations":7, "native american":7, "nativeamerican":7, "native-american":7, "indian american":7, "indianamerican":7, "indian-american":7, "first nations":7, "firstnations":7, "first-nations":7, "indigenous":7, "pacificislander":8, "pacific islander":8, "pacific-islander":8, "southasian":9, "south asian":9, "south-asian":9, "indian":9, "southeastasian":10, "southest asian":10, "southeast asian":10, "southeast-asian":10}
eth_dict_istock = {"Northern European Descent":2, "Scandinavian Descent":2, "Southern European Descent":2, "East Asian Ethnicity":3, "Japanese Ethnicity":3, "Chinese Ethnicity":3, "Southeast Asian Ethnicity":10, "South Asian Ethnicity":9, "West Asian Ethnicity":5, "North African Ethnicity":5, "African-American Ethnicity":1, "Latin American and Hispanic Ethnicity":4, "Cuban Ethnicity":4, "Puerto Rican Ethnicity":4, "Mexican Ethnicity":4, "Multiracial Group":6, "Multiracial Person":6, "Russian Ethnicity":2, "Eastern European Descent":2, "Korean Ethnicity":3,  "Filipino Ethnicity":10, "Vietnamese Ethnicity":10, "Thai Ethnicity":10, "Cambodian Ethnicity":10, "Indian Ethnicity":9, "Sri Lankan Ethnicity":9,  "Italian Ethnicity":2, "East Slavs":2, "Polish Ethnicity":2, "Ukrainian Ethnicity":2, "Spanish and Portuguese Ethnicity":2,  "Chinese Han":3,  "Nepalese Ethnicity":3, "Taiwanese Ethnicity":3, "Only Japanese":3, "Tibetan Ethnicity":3, "Malaysian Ethnicity":10,}
eth_dict_istock_secondary = {"Ethiopian Ethnicity":1, "Southern African Tribe":1, "Maasai People":1, "East African Ethnicity":1, "Western African Peoples":1, "Haitian Ethnicity":1, "Afro-Caribbean Ethnicity":1, "Trinidadian Ethnicity":1, "Creole Ethnicity":1, "Jamaican Ethnicity":1, "Karo Tribe":1, "Nilotic Peoples":1, "Turkana Tribe":1, "Hamer Tribe":1, "Mursi People":1, "Arbore People":1, "Borana Oromo People":1, "Konso - Tribe":1, "Lobi Tribe":1, "Samburu Tribe":1, "Malagasy People":1, "Himba":1, "Herero Tribe":1, "Zulu Tribe":1, "Nuer People":1, "San Peoples":1, "Hadza People":1, "Wodaabe Tribe":1, "Fula People":1, "Indigenous Peoples of Africa":1, "Betsileo Tribe":1, "Tuareg Tribe":1, "Kazakh Ethnicity":3, "Sherpa":3, "Dong Tribe":3, "Dong Tribe":3, "Meo":3, "Hani Tribe":3, "Miao Minority":3, "Monguor":3, "Sherpa":3, "Central Asian Ethnicity":3, "Kyrgiz":3, "Romani People":2,  "Albanian Ethnicity":2, "Israeli Ethnicity":2, "Indigenous Peoples of the Americas":7, "Inuit":7, "Sami People":2, "Métis Ethnicity":7, "Quechua People":7, "Indigenous Peoples of South America":7, "Uros":7, "Argentinian Ethnicity":4, "Ecuadorian Ethnicity":4, "Peruvian Ethnicity":4, "Brazilian Ethnicity":4, "Bolivian Ethnicity":4, "Chilean Ethnicity":4, "Colombian Ethnicity":4, "Venezuelan Ethnicity":4, "South American Ethnicity":4, "Berbers":5, "Egyptian Ethnicity":5, "Armenian Ethnicity":2, "Dominican Ethnicity":6, "Eurasian Ethnicity":6, "Garifuna Ethnicity":6, "Pardo Brazilian":6, "Māori People":7, "Pacific Islanders":7, "Hawaiian Ethnicity":7, "Polynesian Ethnicity":7, "Samoan Ethnicity":7, "Melanesian Ethnicity":7, "Kanak People":7, "Asaro People":7,  "Sinhalese People":9, "Bengali People":9, "Maldivian Ethnicity":9, "Kubu Tribe":10, "Khmer People":10,  "Mongolian Ethnicity":10, "Palaung Tribe":10, "Padaung Tribe":10, "Rawang":10, "Burmese Ethnicity":10, "Akha Tribe":10, "Sea Gypsy":10, "Moken Tribespeople":10, "Malay People":10, "Hill Tribes":10, "Red Zao":10, "Indonesian Ethnicity":10, "Kubu Tribe":10, "Kurdish Ethnicity":5, "Lebanese Ethnicity":5, "Middle Eastern Ethnicity":5, "Bedouin":5, "Pakistani Ethnicity":5, "Iranian Ethnicity":5, "Turkish Ethnicity":5, "Afghan Ethnicity":5, "Pashtuns":5, "Hazara":5, "Baloch":5, "Tajiks Ethnicity":5, "Kalash People":5}
eth_dict_shutter_secondary = {"african american girl":1, "african girl":1, "black girl":1, "black girls":1, "black executive":1, "black leaders":1, "black professional woman":1, "black woman beauty":1, "caucasian girl":2, "eastern european girl":2, "english girl":2, "greek girl":2, "lithuanian girl":2, "polish girl":2, "polish girl in uk":2, "russian girl":2, "swedish girl":2, "white dress girl":2, "asian baby girl":3, "oriental school girl":3, "asian girl":3, "asian girl face":3, "asian girl isolated":3, "asian girls":3, "asian hijab girl":3, "chinese baby girl":3, "chinese girl":3, "east asian girl":3, "girlchinese":3, "japanese baby girl":3, "korean girl":3, "pretty asian girl":3, "brazilian girl":4, "latin girl":4, "latina girl":4, "moroccan girl":5, "biracial girl":6, "mixed race girl":6, "indian girl":9, "indian girl child":9, "indian girl with long hair":9, "pakistani girl":9, "indonesia girl":10, "indonesian girl":10, "malaysia girls":10, "malaysian girl":10, "malaysian girls":10, "thailand girls":10, "young afro woman":1, "african american woman":1, "african american woman smiling":1, "african american woman swimming":1, "african woman":1, "african woman isolated":1, "afro american woman":1, "afro woman":1, "black professional woman":1, "black woman":1, "black woman beauty":1, "black woman face":1, "black woman smiling":1, "stylish african woman":1, "a young white woman":2, "british woman":2, "caucasian woman":2, "caucasian woman beach":2, "caucasian woman on sandy beach":2, "caucasian woman smiling":2, "caucasian-woman":2, "eastern european woman":2, "greek woman":2, "italian woman":2, "lithuanian woman":2, "scandinavian woman":2, "swedish woman":2, "ukrainian woman":2, "asian young woman":3, "asian business woman":3, "asian businesswoman":3, "asian businesswoman with documents":3, "asian businesswoman with globe":3, "asian businesswoman with mobile":3, "asian businesswoman with santa hat":3, "asian businesswoman with tablet":3, "asian businesswoman writing":3, "asian woman":3, "asian woman doctor":3, "asian woman face":3, "asian woman smiling":3, "cute asian woman":3, "japanese woman":3, "nepalese woman":3, "pretty asian woman":3, "telephone asian woman":3, "young hispanic woman nude":4, "latin woman":4, "latino woman":4, "middle eastern woman":4, "biracial woman":6, "trans woman":7, "indian woman":9, "indian woman doctor":9, "indian woman driving":9, "pretty indian woman":9, "south asian woman":9, "indonesian woman":10, "southeast asian woman":10, "black female model":1, "caucasian female":2, "white female":2, "asian female driver":2, "female caucasian":2, "female caucasian color colour":2, "female white":2, "large busted asian female model":3, "hijab female":5, "caucasian baby boy":2, "caucasian boy":2, "english boy":2, "european boy":2, "white boy":2, "asian boy":3, "asian boy face":3, "chinese boy":3, "muslim boy":5, "mix race boy":6, "mixed race boy":6, "african american man":1, "african man":1, "black man":1, "black man face":1, "nigerian man":1, "angry white man":2, "british man":2, "caucasian man":2, "caucasian young man":2, "european man":2, "italian man":2, "white man":2, "white man 3d":2, "white man.":2, "asian man":3, "chinese man":3, "japanese man":3, "brazilian man":4, "hispanic man":4, "latin man":4, "arabian man":5, "muslim man":5, "turkish man":5, "mix race man":6, "mixed race man":6, "indian man":9, "sikh man":9, "south asian man":9, "filipino man":10, "old caucasian male":2, "british male":2, "caucasian male":2, "italian male":2, "male caucasian":2, "white male":2, "white male baby":2, "young white male model":2, "asian male":3, "mixed race male":6}



# load Keywords_202304300930.csv as df, drop all but keytype Locations, create two dicts: string->ID & GettyID->ID  
# loc_dict = {"Canada":1989}

age_dict = {
    "newborn":1,
    "baby":1,
    "infant":2,
    "infants":2,
    "toddlers":3,
    "toddler":3,
    "child":3,
    "children":3,
    "childrens":3,
    "girls":3,
    "boys":3,
    "girl":3,
    "boy":3,
    "jeune":3,
    "junger":3,
    "kinder":3,
    "bambino":3,
    "bambina":3,
    "ragazza":3, 
    "schoolgirls":3, 
    "jungen":3,
    "-women -men children":3,
    "teen":4,
    "teens":4,
    "teenager":4,
    "teenagers":4,
    "young adult":5,
    "young man":5,
    "young woman":5,
    "young men":5,
    "young women":5,
    "young":5,
    "20s":5,
    "30s":5,
    "twenties": 5,
    "thirties": 5,
    "adult":6,
    "40s":6,
    "50s":6,
    "forties": 6,
    "fifties": 6,
    "old":7,
    "60s":7,
    "70s":7,
    "80s":7,
    "90s":7,
    "70+":7,
    "sixties": 7,
    "seventies": 7,
    "eighties": 7,
    "older":7,
    "seniorin":7
}
age_dict_istock = {"0-1 Months":1, "0-11 Months":1, "Babies Only":1, "2-5 Months":2, "6-11 Months":2, "Preschool Age":2, "12-17 Months":3, "2-3 Years":3, "4-5 Years":3, "6-7 Years":3, "8-9 Years":3, "10-11 Years":3, "12-13 Years":3, "12-23 Months":3, "Elementary Age":3, "Pre-Adolescent Child":3, "Children Only":3, "18-23 Months":3, "14-15 Years":4, "16-17 Years":4, "18-19 Years":4, "Teenagers Only":4, "20-24 Years":5, "25-29 Years":5, "30-34 Years":5, "20-29 Years":5, "30-39 Years":5, "35-39 Years":6, "40-44 Years":6, "45-49 Years":6, "50-54 Years":6, "55-59 Years":6, "Adults Only":6, "Mid Adult":6, "Mature Adult":6, "40-49 Years":6, "50-59 Years":6, "60-64 Years":7, "65-69 Years":7, "70-79 Years":7, "Senior Adult":7, "60-69 Years":7, "80-89 Years":7, "Over 100":7, "90 Plus Years":7}
age_dict_shutterstock = {"1 to 2 years":2, "10 to 11 years":4, "10 to 12 years":4, "10 to 13 years":4, "10 years":3, "10 years old":3, "10-12 years":4, "11 years old":4, "12 to 13 years":4, "12 years":4, "12 years old":4, "13 to 14 years":4, "13 to 15 years":4, "13 years old":4, "13-14 years":4, "13-15 years":4, "14 years old":4, "14- 15 years":4, "15 years old":4, "15-16 years":4, "16 to 17 years":4, "16 years old":4, "16-25years":4, "17 years":4, "17 years old":4, "18 to 19 years":4, "18 years old":4, "18-19 years old":4, "19 years":4, "19 years old":4, "19-20 years":4, "2 to 3 years":3, "2 years":3, "2 years old":3, "20 to 24 years":5, "20 to 25 years old":5, "20 years old":5, "20-25 years":5, "20-30 years":5, "21 years":5, "23 years":5, "24-29 years":5, "25 to 29 years":5, "25-28 years":5, "25-30 years":5, "25-30 years old":5, "25-30years":5, "26 years":5, "26-30 years":5, "28-29 years":5, "3 to 4 years":3, "3 years old":3, "3-4 years":3, "30 to 34 years":5, "30-35 years":5, "30-40 years":5, "35 to 39 years":6, "35-30 years":6, "35-40 years":6, "35-40-years":6, "35-45 years":6, "4 to 5 years":3, "4 years old":3, "40 to 44 years":6, "40 to 49 years":6, "40 years old":6, "40-45 years":6, "40-50 years":6, "45 to 49 years":6, "45 years old":6, "45-50 years":6, "48 years":6, "49 years":6, "5 to 6 years":3, "5 to 9 years":3, "5 years":3, "5 years old":3, "5-10 years":3, "5-6 years":3, "5-6 years old":3, "50 to 54 years":6, "50 to 59 years":6, "50-55 years":6, "51 years":6, "55 to 59 years":6, "55-60 years":6, "6 to 7 years":3, "6 years":3, "6 years old":3, "60 to 64 years":7, "60 to 69 years":7, "60-65 years":7, "60-70 years":7, "65 to 69 years":7, "65 years":7, "65-70 years":7, "7 to 8 years":3, "7 to 9 years":3, "7 years":3, "7 years old":3, "7-8 years":3, "7-9 years":3, "70 to 74 years":7, "70 to 79 years":7, "70-75 years":7, "75 to 79 years":7, "78 years":7, "8 to 9 years":3, "8 years":3, "8 years old":3, "80 plus years":7, "80 to 84 years":7, "80-84 years":7, "9 years":3, "9 years old":3, "9-10 years":3, "age 20-25 years":5, "eight years old":3, "eighty years old":7, "fifty years old":6, "four years old":3, "nine years old":3, "seven years old":3, "three years old":3, "two years old":3}
age_dict_shutter_secondary = {"asian baby girl":1, "baby girl fashion":1, "baby girl sitting on a chair":1, "baby-girl":1, "babygirl":1, "beautiful baby girl":1, "beautiful bay girl":1, "gorgeous baby girl":1, "happy baby girl":1, "only one baby girl":1, "premature baby girl":1, "girl age 1":2, "3 year old girl":3, "3 years old girl":3, "4 year old girl":3, "5 year old girl":3, "7 year old girl":3, "9 year old girl":3, "adorable little girl":3, "child girl":3, "children girl":3, "cute little girl":3, "cute little girl eating":3, "cute little girl in a dress":3, "cute young girl":3, "funny little girl eating":3, "litte girl":3, "littl girl":3, "little blonde girl":3, "little girl alone":3, "little girl and her dog":3, "little girl asleep":3, "little girl asleep outdoors":3, "little girl at a rainy window":3, "little girl at ickworth":3, "little girl buried in sand":3, "little girl digging":3, "little girl eating a sausage roll":3, "little girl eating italian food":3, "little girl having fun":3, "little girl hiking":3, "little girl in a red coat":3, "little girl in a white dress":3, "little girl in garden":3, "little girl in the forest":3, "little girl in the snow":3, "little girl in the woods":3, "little girl jaywick":3, "little girl laughing":3, "little girl lost":3, "little girl on beach":3, "little girl paddling":3, "little girl painting":3, "little girl playing":3, "little girl playing in the sea":3, "little girl plying":3, "little girl running":3, "little girl sandcastles":3, "little girl skateboarding":3, "little girl smiling":3, "little girl studying":3, "little girl sunset":3, "little girl tennis":3, "little girl walking":3, "little girl walking by the beach":3, "little girl walking in jaywick":3, "little girl walking in the snow":3, "little girl walking to school in the snow":3, "little girl walking towards a spooky house":3, "little girl wearing armbands in swimming pool":3, "little girl with flower":3, "little happy girl":3, "one little girl":3, "one little girl only":3, "only one pre-adolescent girl":3, "oriental school girl":3, "pre teen girl":3, "pre teen girls":3, "pre-adolescent girl":3, "pre-teen girl":3, "pre-teen girls":3, "preteen girl":3, "preteen girls":3, "pretty girl growing teeth":3, "pretty little girl":3, "pretty young girl in the snow":3, "toddle girl":3, "toddler girl":3, "toddler girl dress":3, "toddler girl playing with magnet board":3, "toddler girl playing with play doh":3, "toddler girls":3, "young girl at the hairdressers":3, "young girl celebrates in studio":3, "young girl climbing":3, "young girl fashion":3, "young girl hiking":3, "young girl in the lake":3, "young girl in woods":3, "young girl looking out to the city":3, "young girl lying down":3, "young girl outdoors":3, "young girl playing":3, "young girl playing outdoors":3, "young girl portrait":3, "young girl running":3, "young girl smiling":3, "young girl wearing mask":3, "young girl.":3, "happy teenage girl":4, "high school girl":4, "one teenage girl":4, "only one teenage girl":4, "teen girl":4, "teenage girl face":4, "teenage girl portrait":4, "teenaged girl":4, "teenaged girls":4, "teenager girl":4, "young teenage girl":4, "college girl":5, "lonely senior woman":4, "20s woman":5, "30s woman":5, "30s woman portrait":5, "a young white woman":5, "asian young woman":5, "attractive young woman":5, "charming young woman":5, "one young adult woman only":5, "one young woman":5, "only one young woman":5, "very attractive young woman":5, "young adult adult woman":5, "young adult woman":5, "young adultwoman":5, "young afro woman":5, "young attractive woman":5, "young attractive woman in red lingerie":5, "young brown hair woman":5, "young business woman isolated":5, "young businesswoman":5, "young hispanic woman nude":5, "young naked woman":5, "young nude woman":5, "young pretty woman":5, "young woman cartwheeling":5, "young woman doing yoga":5, "young woman face":5, "young woman happy":5, "young woman isolated":5, "young woman meditating":5, "young woman modern":5, "young woman portrait":5, "young woman reading":5, "young woman sitting":5, "young woman sitting on floor":5, "young woman smiling":5, "young woman standing":5, "young woman thinking":5, "young woman ubud temple":5, "young woman washing car":5, "young woman white background":5, "young-woman":5, "40 something woman":6, "40s woman":6, "50s woman":6, "55 year old woman":6, "female woman 40s forties comforting profession occupation career":6, "fifties woman":6, "mid adult woman":6, "mid aged woman":6, "mid-adult woman":6, "middle age woman":6, "middle aged woman":6, "middle aged woman isolated":6, "middle aged woman portrait":6, "middle-aged woman":6, "one mid adult woman":6, "one mid-adult woman only":6, "one middle aged woman only":6, "only one mid adult woman":6, "aged woman":7, "aging woman":7, "attractive older woman":7, "beautiful older woman":7, "beautiful senior woman":7, "elder woman":7, "elderly woman isolated":7, "elderly woman on phone":7, "independent woman senior":7, "mature sexy woman":7, "older woman":7, "older woman face":7, "older woman in a park":7, "older woman portrait":7, "older woman smiling":7, "one senior woman":7, "only one mature woman":7, "only one senior woman":7, "retired woman":7, "senior adult woman":7, "senior businesswoman":7, "senior woman drink tea leisure":7, "senior woman only":7, "seniorwoman":7, "woman 70s":7, "female child":3, "female children":3, "young female model":5, "female young adult":5, "female 20s":5, "young adult female":5, "20s female":5, "20s2 female":5, "30s female":5, "female adult 20-25 years":5, "female in her twenties":5, "female30s":5, "female adult":6, "female adults":6, "middle aged female":6, "mature female":7, "mature female model":7, "senior female":7, "female senior":7, "cute baby boy":1, "only one baby boy":1, "preschool boy":2, "3 years old boy":3, "4 year old boy":3, "5 year old boy":3, "7 year old boy":3, "8 year old boy":3, "active little boy":3, "boy aged 5":3, "boyhood":3, "child boy":3, "children boy":3, "cropped picture of young boy":3, "cute young boy":3, "handsome little boy":3, "little boy asleep on beach":3, "little boy climbing":3, "little boy climbing a rope":3, "little boy colouring":3, "little boy having fun":3, "little boy in woods":3, "little boy isolated":3, "little boy laughing":3, "little boy on a swing":3, "little boy on beach":3, "little boy playing":3, "little boy portrait":3, "little boy sitting":3, "little boy sitting in a pub":3, "little boy with adhd":3, "little boys":3, "little boys only":3, "little boys woods":3, "littlt boy":3, "one little boy":3, "one little boy only":3, "only one pre-adolescent boy":3, "photo of a little boy":3, "portrait of a little boy":3, "pre-adolescent boy":3, "pre-teen boy":3, "school boy":3, "six year old boy":3, "smiling little boy":3, "toddler boy":3, "toddler boy isolated":3, "toddler boys":3, "young boy angry":3, "young boy eating chocolate":3, "young boy face":3, "young boy isolated":3, "young boy listening to headphones":3, "young boy listening to music":3, "young boy playing":3, "young boy portrait":3, "young boy reading":3, "young boy smiling":3, "17 year old boy":4, "adolescent boy":4, "boy teenage":4, "high school boy":4, "one teenage boy":4, "only one teenage boy":4, "teenage boy portrait":4, "teenage boy smiling":4, "teenaged boy":4, "teenaged boys":4, "teenager boy":4, "teenager boys":4, "20s man":5, "30 year old man":5, "a yong man watching the sea":5, "a young man":5, "attractive older man":5, "good looking young man":5, "handsome young man":5, "man age 30":5, "young adult man":5, "young business man":5, "young handsome man":5, "young man backpack":5, "young man deep in thought":5, "young man in city":5, "young man isolated":5, "young man isolated on white":5, "young man moving":5, "young man portrait":5, "young man posing":5, "young man sitting":5, "young man sitting on floor":5, "young man smiling":5, "young man standing":5, "young man walking":5, "young man with backpack and cap":5, "young man with cap":5, "young man working with laptop":5, "young young man":5, "young-adult man":5, "younger man":5, "adult man":6, "man 50 years":6, "man in 40's":6, "man in his 40's":6, "mid adult man":6, "mid-adult man":6, "middle age man":6, "middle aged man":6, "middle aged man smiling":6, "middle aged man standing":6, "middle eastern man":6, "middle-aged man":6, "attractive young man":7, "elderly happy man":7, "kind face old man":7, "man aged 60":7, "mature adult man":7, "real life old man working":7, "retired man":7, "retired man working":7, "senior man portrait":7, "the little old man":7, "the old man of storr":7, "well being elderly man":7, "baby male":1, "toddler male":2, "male child":3, "15 year old male":4, "male 20s":5, "male young adult":5, "one young male only":5, "young adult male":5, "adult male":6, "male adult":6, "male adults":6, "mid-adult male":6, "middle aged male":6, "one mid adult male only":6, "male 70s":7, "male senior":7, "male senior adult":7, "male senior adults":7, "old caucasian male":7, "old male":7, "older male":7, "senior male":7, "one young adult man":5, "one young man":5, "only one young man":5, "one mid-adult man only":6, "only one mid adult man":6, "old man and cat":7, "old man and dog":7, "old man and pets":7, "old man face":7, "old man holding spanner in workshop":7, "old man in red hat working":7, "old man of coniston":7, "old man point":7, "old man portrait":7, "old man singing":7, "old man smiling":7, "old man smiling while working":7, "old man storr":7, "old man working in workshop":7, "old-man":7, "older man":7, "older man portrait":7, "one mature man":7, "one senior man":7, "only one mature man":7, "only one senior man":7, "only senior man":7}

age_details_dict = {
    'toddler': 2,
    '20s': 4,
    '30s': 5,
    '40s': 6,
    '50s': 7,
    '60s': 8,
    '70+': 9
}
age_detail_dict_istock = {"0-1 Months":1, "2-5 Months":1, "0-11 Months":1, "Babies Only":1, "12-17 Months":2, "2-3 Years":2, "6-11 Months":2, "12-23 Months":2, "18-23 Months":2, "12-13 Years":3, "14-15 Years":3, "16-17 Years":3, "18-19 Years":3, "20-24 Years":4, "25-29 Years":4, "20-29 Years":4, "30-34 Years":5, "35-39 Years":5, "30-39 Years":5, "40-44 Years":6, "45-49 Years":6, "40-49 Years":6, "50-54 Years":7, "55-59 Years":7, "50-59 Years":7, "60-64 Years":8, "65-69 Years":8, "60-69 Years":8, "70-79 Years":9, "80-89 Years":9, "Over 100":9, "90 Plus Years":9, "4-5 Years":10, "6-7 Years":10, "8-9 Years":11, "10-11 Years":11}
age_details_dict_shutterstock = {"20 to 24 years":4, "20 to 25 years old":4, "20 years old":4, "20-25 years":4, "20-30 years":4, "21 years":4, "23 years":4, "24-29 years":4, "25 to 29 years":4, "25-28 years":4, "25-30 years":4, "25-30 years old":4, "25-30years":4, "26 years":4, "26-30 years":4, "28-29 years":4, "30 to 34 years":5, "30-35 years":5, "30-40 years":5, "35 to 39 years":5, "35-30 years":5, "35-40 years":5, "35-40-years":5, "35-45 years":5, "40 to 44 years":6, "40 to 49 years":6, "40 years old":6, "40-45 years":6, "40-50 years":6, "45 to 49 years":6, "45 years old":6, "45-50 years":6, "48 years":6, "49 years":6, "50 to 54 years":7, "50 to 59 years":7, "50-55 years":7, "51 years":7, "55 to 59 years":7, "55-60 years":7, "60 to 64 years":8, "60 to 69 years":8, "60-65 years":8, "60-70 years":8, "65 to 69 years":8, "65 years":8, "65-70 years":8, "70 to 74 years":9, "70 to 79 years":9, "70-75 years":9, "75 to 79 years":9, "78 years":9, "80 plus years":9, "80 to 84 years":9, "80-84 years":9, "age 20-25 years":4, "eighty years old":9, "fifty years old":7}

def lower_dict(this_dict):
    lower_dict = {k.lower(): v for k, v in this_dict.items()}
    return lower_dict

gender_dict = lower_dict({**gender_dict, **gender_dict_istock, **gender_dict_sex, **gender_dict_sexplural, **gender_dict_both})
gender_dict_secondary = lower_dict({**gender_dict, **gender_dict_shutter_secondary})
eth_dict = lower_dict({**eth_dict, **eth_dict_istock})
age_dict = lower_dict({**age_dict, **age_dict_istock, **age_dict_shutterstock})
age_dict_secondary = lower_dict({**age_dict, **age_dict_shutter_secondary})
age_details_dict = lower_dict({**age_details_dict, **age_detail_dict_istock, **age_details_dict_shutterstock, **age_dict_shutter_secondary})
eth_dict_secondary = lower_dict({**eth_dict_istock_secondary, **eth_dict_shutter_secondary})

# for searching descrption for eth keywords, get rid of ambiguous/polyvalent terms
eth_keys_dict = eth_dict
for k in ['black', 'african']: eth_keys_dict.pop(k)

# table_search ="Images i JOIN ImagesKeywords ik ON i.image_id = ik.image_id JOIN Keywords k on ik.keyword_id = k.keyword_id"
SELECT = "DISTINCT(i.image_id), i.gender_id, author, caption, contentUrl, description, imagename"
FROM ="Images i JOIN ImagesKeywords ik ON i.image_id = ik.image_id JOIN Keywords k on ik.keyword_id = k.keyword_id LEFT JOIN Encodings e ON i.image_id = e.image_id "
WHERE = "e.image_id IS NULL"
LIMIT = 10


# engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
#                                 .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)

engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
), poolclass=NullPool)

# metadata = MetaData(engine)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()


PEXELS_HEADERS = ["id", "title", "keywords", "country", "number_of_people", "orientation", "age","gender", "ethnicity", "mood", "image_url", "image_filename"]
ONETWOTHREE_HEADERS = ["id","title","keywords","orientation","age","word","exclude","people","ethnicity","image_url","image_filename"]
KEYWORD_HEADERS = ["keyword_id","keyword_number","keyword_text","keytype","weight","parent_keyword_id","parent_keyword_text"]
IMG_KEYWORD_HEADERS = ["site_image_id","keyword_text"]
header_list = PEXELS_HEADERS

def read_csv(csv_file):
    with open(csv_file, encoding="utf-8", newline="") as in_file:
        reader = csv.reader(in_file, delimiter=",")
        next(reader)  # Header row

        for row in reader:
            yield row

def write_csv(path,value_list):

    # headers = header_list

    with open(path, 'a') as csvfile: 
    # with open('lat_lon', 'w') as csvfile:
        writer=csv.writer(csvfile, delimiter=',')
        writer.writerow(value_list)

def init_csv(path, headers):
    csvpath_isfile = pathlib.Path(path)
    if not csvpath_isfile.is_file():
        write_csv(path, headers)


def nan2none(this_dict):
    for key, value in this_dict.items():
        if isinstance(value, float) and pd.isnull(value):
            this_dict[key] = None
    return this_dict

def make_key_dict_col3(filepath):
    keys = read_csv(filepath)
    keys_dict = {}
    for row in keys:
        keys_dict[row[2].lower()] = row[0]

    # do I need to pop header? or can I just leave it. 
    return keys_dict

def make_key_dict_col7(filepath):
    keys = read_csv(filepath)
    keys_dict = {}
    for row in keys:
        keys_dict[row[7].lower()] = row[0]

    # do I need to pop header? or can I just leave it. 
    return keys_dict

def make_key_dict(filepath, keytype=None):
    keys = read_csv(filepath)
    keys_dict = {}
    for row in keys:
        if keytype is None:
            keys_dict[row[2].lower()] = row[0]
        elif keytype and row[3].lower() == keytype:
            # print("gotta keytype")
            keys_dict[row[3].lower()] = row[0]
        else:
            # if not none, but not the specific keytype
            pass
            # print("nothing here")

    # do I need to pop header? or can I just leave it. 
    return keys_dict

def get_counter():
    # read last completed file
    try:
        print("trying to get last saved")
        with open(CSV_COUNTOUT_PATH, "r") as f1:
            last_line = f1.readlines()[-1]
        # last_line = read_csv(CSV_COUNTOUT_PATH)
        # print(type(last_line))
        start_counter = int(last_line)
    except:
        start_counter = 0
        print('[download_images_from_cache] set max_element to 0 \n', flush=True)
        print("max_element,", start_counter)
    return start_counter

# returns one mode value from list, ignoring None values
def get_mode(data):
    filtered_data = [value for value in data if value is not None]
    
    # Check if the filtered_data is empty before proceeding
    if not filtered_data:
        return None  # or return any other default value you prefer
    
    mode_counts = Counter(filtered_data)
    mode_value, max_count = mode_counts.most_common(1)[0]
    return mode_value



# abstraction to run a list through unlock_key_plurals_etc
def unlock_key_list(site_image_id, keys_list, keys_dict):
    key_nos_list = []
    
    for key in keys_list:
        key_no = unlock_key_plurals_etc(site_image_id.lower(), key, keys_dict)
        # print(key_no)
        if key_no:
            key_nos_list.append(key_no)
    return key_nos_list

# takes a key and runs all permutations through the dict, and saves missing ones
# this is the kitchen sink function
def unlock_key_plurals_etc(site_id,key, this_dict):
    key_no = None
    key = key.lower()
    try:
        # print("trying basic keys_dict for this key:")
        # print(key)
        # print("from dict this long")
        # print(len(this_dict))
        # # print(this_dict)

        # print(this_dict["office"])
        key_no = this_dict[key]
        # print("this is the key_no")
        # print(key_no)
        return(key_no)
    except:
        try:
            key = key2key[key]
            key_no = this_dict[key]
            return(key_no)
        except:
            # try:
            # try with inflections
            # print(site_id)
            # print(key)

            plur_key = getInflection(key, 'NNS')
            sing_key = getInflection(key, 'NN')
            gerund_key = getInflection(key, 'VBG')
            # print("inflected are: ", plur_key, sing_key, gerund_key)
            if plur_key and key != plur_key:
                try:
                    key_no = this_dict[plur_key[0]]
                    # key = plur_key
                    print(key_no)
                except:
                    pass# print(key)
            elif sing_key and key != sing_key:
                try:
                    key_no = this_dict[sing_key[0]]
                    # key = plur_key
                    # print(key_no)
                except:
                    pass
                    # print(key)
            
            if gerund_key and key != gerund_key:
                try:
                    key_no = this_dict[gerund_key[0]]
                    # key = plur_key
                    # print("gotta key_no, " ,key_no)
                except:
                    pass
                    # print(key)

            if pd.isnull(key_no):
                # if nothing worked, save key, but only if site_id > 10
                # for gender/age, it passes in site_name_id, not site_image_id
                if not isinstance(site_id, int) and not key.startswith('-'):
                    # print(type(site_id))
                    # print(site_id)
                    value_list = [site_id,key]
                    # print("value_list")
                    # print(value_list)
                    write_csv(CSV_NOKEYS_PATH,value_list)
                # print(value_list)
                return
            else:
                value_list = [site_id,key_no]
                write_csv(CSV_IMAGEKEYS_PATH,value_list)
                return key_no


# finds dict items in a string
# called by search_keys
def findall_dict(my_dict,description):
    # Create a regular expression pattern that matches complete words in the dictionary, ignoring case
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(key) for key in my_dict.keys()) + r')\b', re.IGNORECASE)

    # # this re is greedier, matching partial words, w/o boundaries
    # pattern = re.compile(r'\b(' + '|'.join(my_dict.keys()) + r')\b', re.IGNORECASE)


    # Use the regular expression pattern to search for matches in the given string
    matches = pattern.findall(description)

    # Print the values associated with the matching keys
    for match in matches:
        value = my_dict.get(match.lower())
        if value is not None:
            return(value)
            # gender = value
            # print(f'The color of {match} is {value}')

# only searches through keys with dict
# called in by the get_eth_dictonly
def search_keys(keys_list, this_dict, do_write_csv, multi=False):
    results = []
    found_eth2 = False
    for key in keys_list:
        # found = findall_dict(this_dict,key)
        try:
            found = unlock_key_dict(key, this_dict)
        except:
            found = None
        if found is not None:
            found_eth2 = True
            results.append(found)
            print('search_keys found:', found,"from key:", key)
            if do_write_csv:
                write_csv(CSV_ETH2_PATH,[key,found])
    if found_eth2 and do_write_csv: 
        write_csv(CSV_ETH2_PATH,keys_list)
            #age needs to be int()
            # print(results)

        # try:
        #     found = unlock_key_dict(key, this_dict)
        # # this is searching the keys based on the dict, note unlocking dict with keys
        # # found = findall_dict(this_dict,key)
        #     if found is not None:
        #         results.append(found)
        #         print('found it in keywords:', found,"from key:", key)
        # #     #age needs to be int()
        # #     # print(results)
        # except OperationalError as e:
        #         print("exception on unlock_key_dict")
        #         print(e)


    if len(set(results)) == 1:
        one_result = int(results[0])
        # print("found a GOOD result: ", one_result)
    else:
        one_result = 0
        # print("failed search: ", one_result)

    # returns one or many, in a list
    if multi:
        results_list = list(set(results))
    else:
        results_list = [one_result]
    return results_list

# print(key_nos_list)

def get_key_no_dictonly(eth_name, keys_list, this_dict, do_write_csv=False):
    # eth_name = df['ethnicity'][ind]
    # print('isnan?')
    # print(np.isnan(eth_name))
    key_no_list = []
    key_no = None
    # if eth_name is not None or eth_name is not np.isnan(eth_name):
    if not pd.isnull(eth_name):
        try:
            key_no = unlock_key_dict(eth_name, this_dict)
            # key_no = eth_dict[eth_name.lower()]
        # need to key this into integer, like with keys
            # print("eth_name ",eth_name)
        except:
            key_no = None
            print("eth_dict failed with this key: ", eth_name)
        key_no_list.append(key_no)
    else:
        key_no_list = search_keys(keys_list, this_dict, do_write_csv, True)
        print("searched keys and found key_no: ", key_no_list)
    return(key_no_list)

def unlock_key_dict(key,this_dict,this_key2key=None):
    key_no = None
    key = key.lower()
    try:
        try:
            key_no = this_dict[key]
            print(f"unlock_key_dict yields key_no {str(key_no)} for {key}")
            return(key_no)
        except:
            # try again without underscores
            key_no = this_dict[key.replace("_"," ")]
            print(f"unlock_key_dict without underscores yields key_no {str(key_no)} for {key}")
            return(key_no)            
    except:
        if this_key2key:
            try:
                altkey = this_key2key[key.lower()]
                print("altkey")
                print(altkey)
                key_no = this_dict[altkey.lower()]
                print("this is the key_no via loc2loc")
                print(key_no)
                return(key_no)
            except:
                try:
                    #hard coding the second location dict here
                    key_no = locations_dict_alt[key.lower()]
                except:
                    print("NEW KEY -------------------------> ", key)
                    write_csv(CSV_NOLOC_PATH,[key])
                    return(999999999)
        else:
            pass
            # print out for testing purposes
            # print("unlock_key_dict failed, and no key2key for this key: ", key)

# def get_location(df, ind, keys_list):
#     location = None
#     key = df['country'][ind]
#     if not pd.isnull(key):
#         try:
#             location = loc_dict[key]
#         except:
#             print('NEW KEY, NOT IN COUNTRY -------------------------> ', key)
#     # else:
#     #     print('NULL country: ', key)

#     return(location)


#currently only called for keys to desc in main show. not for gender.
def description_to_keys(description, site_id, this_dict="keys_dict"):
    if this_dict=="keys_dict":
        this_dict = keys_dict

    description = description.replace(",","").replace("'s","").replace(".","")

    # print("description_to_keys")    
    key_nos_list =[]
    # description = df['title'][ind]
    # print(description)
    key_no = None
    desc_keys = description.split(" ")
    # print("desc_keys ",desc_keys)
    for key in desc_keys:
        # print("checking key ", key)
        if not pd.isnull(key):
            key_no = unlock_key_plurals_etc(site_id,key,this_dict)
            # print("key_no passed through:")
            # print(key_no)
            if key_no:
                key_nos_list.append(key_no)
            # print("key_nos_list ",key_nos_list)
    return key_nos_list



def get_gender_age_row(gender_string, age_string, description, keys_list, site_id):
    global gender_dict
    global age_dict
    global age_details_dict
    global skip_keys

    def try_gender_age_key(gender, age, age_detail, this_string, extra_dict=False):
        global gender_dict
        global age_dict
        global age_details_dict
        if extra_dict:
                gender_dict = gender_dict_secondary
                age_dict = age_dict_secondary
        if not pd.isnull(this_string):
            # print(f"looking for {this_string} in dict")
            #convertkeys
            try:
                gender = gender_dict[this_string.lower()]
                if gender == 3:
                    gender = 1
                    age = 7
                elif gender == 4:
                    gender = 8
                    age = 7
                elif gender == 9:
                    gender = 1
                    age = 5
                elif gender == 10:
                    gender = 8
                    age = 5
                print(f"first try, gender is {str(gender)} and age is {str(age)}")
                # gender_dict={"men":1, "none":2, "oldmen":3, "oldwomen":4, "nonbinary":5, "other":6, "trans":7, "women":8, "youngmen":9, "youngwomen":10}
            except:
                try:
                    age = age_dict[this_string.lower()]
                    print(f"second try age is {str(age)}")

                except:
                    # this isn't relevant for iStock
                    # commenting out
                    # print('NEW KEY, NOT AGE OR GENDER -------------------------> ', this_string)
                    pass
            try:
                age_detail = age_details_dict[this_string.lower()]
            except:
                # no age detail. it is only on some sites
                pass

        # else:
        #     print("string is None")
        return gender, age, age_detail

    def prioritize_age_gender(gender_list,age_list):
        print(f"prioritize_age_gender gender_list is {gender_list}")
        if (1 in gender_list and 8 in gender_list) or (3 in gender_list and 4 in gender_list) or (9 in gender_list and 10 in gender_list):
            gender = 11
            age = get_mode(age_list)
        elif gender_list.count(3) > 0:
            gender = 1
            age = 7
        elif gender_list.count(4) > 0:
            gender = 8
            age = 7
        elif gender_list.count(9) > 0:
            gender = 1
            age = 5
        elif gender_list.count(10) > 0:
            gender = 8
            age = 5
        else:
            gender = get_mode(gender_list)
            age = get_mode(age_list)
        return gender, age

    def get_gender_age_keywords(gender, age, age_detail, keys_list):
        global skip_keys
        gender_list=[]
        age_list=[]
        age_detail_list=[]
        for key in keys_list:
            # print("key is ", key)
            # reset variables
            if key in skip_keys:
                print("skip_keys for other")
                continue
            gender = None
            age= None
            age_detail= None
            gender, age, age_detail = try_gender_age_key(gender, age, age_detail, key, extra_dict=True)
            gender_list.append(gender)
            age_list.append(age)
            age_detail_list.append(age_detail)
        # print(gender_list)
        # print(age_list)
        # print(age_detail_list)
        gender, age = prioritize_age_gender(gender_list,age_list)
        age_detail = get_mode(age_detail_list)
        return gender, age, age_detail

    print("get_gender_age_row starting")
    global gender_dict
    gender = None
    age= None
    age_detail= None

    description = description.replace(",","").replace("'s","").replace(".","")

    # print("gender_string, age_string",gender_string, age_string)
    # print("types",type(gender_string), type(age_string))

    # this if/if structure is necessary because "" and isnull were not compatible
    # Get gender
    # why isn't this working right? 
    if gender_string != "":
        print("trying try_gender_age_key for", gender_string, age_string)
        gender, age, age_detail = try_gender_age_key(gender, age, age_detail, gender_string)
        print(gender)
        print(age)
        print(age_detail)

    else:
        print("trying get_gender_age_keywords for", gender_string, age_string)
        gender, age, age_detail = get_gender_age_keywords(gender, age, age_detail, keys_list)
        print(gender)
        print(age)
        print(age_detail)


        #try keys for gender
    # print("gender, age, after try key gender_string")
    # print(gender)
    # print(age)
    if pd.isnull(gender): 
        # print("gender is null")
        gender_results = description_to_keys(description, site_id, gender_dict)
        if gender_results and len(set(gender_results)) == 1:
            gender = gender_results[0]
        elif len(set(gender_results)) > 1: 
            for result in gender_results:
                if 8 and 10 in gender_results:
                    print("GOTTA TEEN F", gender_results)
                    gender = 8
                    desc_age = 5
                elif 1 and 9 in gender_results:
                    print("GOTTA TEEN M", gender_results)
                    gender = 1
                    desc_age = 5
                elif 8 and 4 in gender_results:
                    print("GOTTA OLD F", gender_results)
                    gender = 8
                    desc_age = 7
                elif 1 and 3 in gender_results:
                    print("GOTTA OLD M", gender_results)
                    gender = 1
                    desc_age = 7
                else:
                    print("too many genders", gender_results)
    # print("gender, age, after try description gender_string")
    # print(gender)
    # print(age)

    # Get age 
    if age_string != "":
        gender, age, age_detail = try_gender_age_key(gender, age, age_detail, age_string)
        #this shouldn't overwrite the existing gender if it is not None
    # print("gender, age, after try key age_string")
    # print(gender)
    # print(age)
    if pd.isnull(age):
        print("age is still null, trying keywordsa again, but not for gender")
        _, age, age_detail = get_gender_age_keywords(gender, age, age_detail, keys_list)
        print(gender)
        print(age)
        print(age_detail)
        if pd.isnull(age):
            print("age is really still null, trying description")

            age_results = description_to_keys(description, site_id, age_dict)
            if len(set(age_results)) == 1:
                age = age_results[0]


    print("gender, age, after everything")
    print(gender)
    print(age)

    if not age or not gender:
        print("MISSING AGE OR GENDER, IS IT IN THE KEYS?", keys_list)

    return gender, age, age_detail


'''
1   getty
2   shutterstock
3   adobe
4   istock
5   pexels
6   unsplash
7   pond5
8   123rf
9   alamy
10  visualchinagroup
'''

def get_hash_folders(filename):
    m = hashlib.md5()
    m.update(filename.encode('utf-8'))
    d = m.hexdigest()
    # print(d)
    # csvWriter1.writerow(["https://upload.wikimedia.org/wikipedia/commons/"+d[0]+'/'+d[0:2]+'/'+filename])
    return d[0].upper(), d[0:2].upper()

def generate_local_unhashed_image_filepath(image_name):
    file_name_path = image_name.split('?')[0]
    file_name = file_name_path.split('/')[-1].replace(".jpeg",".jpg")
    # extension = file_name.split('.')[-1]
    hash_folder, hash_subfolder = get_hash_folders(file_name)
    # print("hash_folder: ",hash_folder)
    # print("hash_subfolder: ", hash_subfolder)
    print (os.path.join(hash_folder, hash_subfolder,file_name))
    return os.path.join(hash_folder, hash_subfolder,file_name)
        # IMAGES_FOLDER_NAME, hash_folder, '{}.{}'.format(file_name, extension))

# I don't think there is a df? 
def structure_row_pexels(df, ind, keys_list): 
    site_id = 5
    gender_key, age_key = get_gender_age(df, ind, keys_list, site_id)
    if 'country' in df.columns:
        location_no = get_location(df, ind, keys_list)
    else:
        location_no = None
    image_row = {
        "site_image_id": df.loc[ind,'id'],
        "site_name_id": site_id,
        "description": df['title'][ind],
        "location_id": location_no,
        # "": df['number_of_people'][ind] # should always be one. If not one, toss it? 
        # "": df['orientation'][ind]
        "age_id": age_key,
        "gender_id": gender_key,  
        # "location_id":"0",
        # "": df['mood'][ind]
        "contentUrl": df['image_url'][ind],
        "imagename": generate_local_unhashed_image_filepath(df['image_url'][ind]) # need to refactor this from the contentURL using the hash function
    }
    return(nan2none(image_row))


def structure_row_123_asrow(row, ind, keys_list):
    site_id = 8
    gender = None
    age = row[4]
    description = row[1]
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, site_id)

    image_row = {
        "site_image_id": row[0],
        "site_name_id": site_id,
        "description": description[:140],
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": row[9],
        "imagename": generate_local_unhashed_image_filepath(row[9])  # need to refactor this from the contentURL using the hash function
    }
    
    return nan2none(image_row)


def structure_row_adobe(row, ind, keys_list):
    site_id = 3
    gender = row[7]
    age = None
    description = row[1]
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, site_id)

    image_row = {
        "site_image_id": row[0],
        "site_name_id": site_id,
        "description": description[:140],
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": row[8],
        "imagename": generate_local_unhashed_image_filepath(row[9].replace("images/",""))  # need to refactor this from the contentURL using the hash function
    }
    
    return nan2none(image_row)


def structure_row_istock(row, ind, keys_list):
    print(row[11])
    site_id = 4 #id for the site, not the image
    gender = row[7].replace("_"," ")
    age = row[6].replace("_"," ")
    country = row[3].replace("_"," ")
    description = row[1]
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, site_id)
    country_key = None
    print("row 3 is", type(country))
    if country and country != "":
        country_key = unlock_key_dict(country,locations_dict, loc2loc)
        if country_key == 999999999:
            write_csv(CSV_BLANKLOC_PATH,row)
            return(None)
    if not country or country == "":
        #tk search keys
        # get eth from keywords, using keys_list and eth_keys_dict
        print("UNLOCKING SEARCH_KEYS_FOR_LOC <><><><><><><><>")
        # absence of search string ("None") triggers search_keys function
        loc_no_list = get_key_no_dictonly(None, keys_list, locations_dict, True)
        print(loc_no_list)
        country_key = get_mode(loc_no_list)
        # if not loc_no_list:
        #     loc_no_list = get_key_no_dictonly(None, keys_list, locations_dict_alt, True)
        #     if loc_no_list: 
        print(f"SEARCH_KEYS_FOR_LOC found for {country_key}")
    else:
        country_key = None

    # handle long URLs
    if len(row[10])>300: contentUrl = "contentUrl was greater than 300 characters search for site_image_id"
    else: contentUrl = row[10]

    image_row = {
        "location_id": country_key,        
        "site_image_id": row[0],
        "site_name_id": site_id,
        "description": description[:140],
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": contentUrl,
        "imagename": generate_local_unhashed_image_filepath(row[11].replace("images/","").split("?")[0])  # need to refactor this from the contentURL using the hash function
    }
    
    return nan2none(image_row)

def structure_row_shutterstock(row, ind, keys_list):
    # print(row[11])
    site_id = 2 #id for the site, not the image
    gender = row[6].replace("_"," ")
    age = row[5].replace("_"," ")
    country = row[3].replace("_"," ")
    description = row[1]
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, site_id)
    country_key = None
    print("row 3 is", type(country))
    if country and country != "":
        country_key = unlock_key_dict(country,locations_dict_AA, loc2loc)
        # print("country_key", str(country_key))
        if country_key == 999999999:
            write_csv(CSV_BLANKLOC_PATH,row)
            return(None)
    elif not country or country == "":
        #tk search keys
        # get eth from keywords, using keys_list and eth_keys_dict
        print("UNLOCKING SEARCH_KEYS_FOR_LOC <><><><><><><><>")
        # absence of search string ("None") triggers search_keys function
        loc_no_list = get_key_no_dictonly(None, keys_list, locations_dict, True)
        print(loc_no_list)
        country_key = get_mode(loc_no_list)
        # if not loc_no_list:
        #     loc_no_list = get_key_no_dictonly(None, keys_list, locations_dict_alt, True)
        #     if loc_no_list: 
        print(f"SEARCH_KEYS_FOR_LOC found for {country_key}")
    else:
        country_key = None

    # handle long URLs
    if len(row[8])>300: contentUrl = "contentUrl was greater than 300 characters search for site_image_id"
    else: contentUrl = row[8]

    image_row = {
        "location_id": country_key,        
        "site_image_id": row[0],
        "site_name_id": site_id,
        "description": description[:140],
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": contentUrl,
        "imagename": generate_local_unhashed_image_filepath(row[9].replace("images/","").split("?")[0])  # need to refactor this from the contentURL using the hash function
    }
    # print(image_row)
    return nan2none(image_row)


# Define a custom retry decorator
def custom_retry(func):
    @retry(
        wait_fixed=1000,  # 1 second wait
        stop_max_attempt_number=5,  # 5 retries
        retry_on_exception=lambda ex: isinstance(ex, OperationalError),
    )
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# Example usage of the custom_retry decorator
@custom_retry
def execute_query_with_retry(conn, query, parameters=None):
    try:
        if parameters:
            result = conn.execute(query, parameters)
        else:
            result = conn.execute(query)
        return result
    except OperationalError as e:
        print(f"OperationalError occurred: {e}")
        raise e

def ingest_csv():

    # change this for each site ingested #
    # adobe
    # column_keys = 6 #where the keywords are
    # separator_keys = " " #for keywords, in the column listed above
    # column_site = 8 #not sure this is used
    # column_eth = None #ethnicity
    # search_desc_for_keys = True

    # # istock
    column_keys = 2 #where the keywords are
    separator_keys = "|" #for keywords, in the column listed above
    # column_site = 8 #not sure this is used
    column_eth = 7 #ethnicity
    search_desc_for_keys = False


    with open(CSV_IN_PATH) as file_obj:
        # reader = csv.reader((row.replace('\0', '').replace('\x00', '') for row in in_file), delimiter=",")

        reader_obj = csv.reader(file_obj)
        next(reader_obj)  # Skip header row
        start_counter = get_counter()
        print("start_counter: ", start_counter)
        # start_counter = 0 #temporary for testing, while adobe is ongoing
        counter = 0
        ind = 0
        
        for row in reader_obj:
            # print(row[1])
            
            if counter < start_counter:
                counter += 1
                continue
            
            # splitting keys. try is for getty, except is currently set for pexels.
            try:
                keys_list = row[column_keys].lower().split(separator_keys)

                # # adobe specific. remove for future sites
                # desc_list = row[3].replace(",","").lower().split("-")
                # keys_list = list(filter(None, keys_list + desc_list))

            except IndexError:
                print("keys failed")
            print("keys_list")
            print(keys_list)


            # image_row = structure_row_adobe(row, ind, keys_list)
            image_row = structure_row_shutterstock(row, ind, keys_list)

            # if the image row has problems, skip it (structure_row saved it to csv)
            if not image_row:
                continue
            
            site_image_id = image_row['site_image_id']

            # get keywords
            key_nos_list = unlock_key_list(site_image_id, keys_list, keys_dict)

            if search_desc_for_keys == True:
                desc_key_nos_list = description_to_keys(image_row['description'], image_row['site_image_id'])
                key_nos_list = set(key_nos_list + desc_key_nos_list)
            

            # this isn't working. not catching nulls. 
            if not pd.isnull(row[column_eth]) and len(row[column_eth])>0:
                print("have eth", row[column_eth].lower(), "type is", type(row[column_eth].lower()))
                eth_no_list = get_key_no_dictonly(row[column_eth].lower(), keys_list, eth_dict)
            else:
                # get eth from keywords, using keys_list and eth_keys_dict
                print("UNLOCKING KEYS FOR eth_keys_dict <><><><><><><><>")
                # absence of search string ("None") triggers search_keys function
                eth_no_list = get_key_no_dictonly(None, keys_list, eth_keys_dict)
                print(eth_no_list)
                if not eth_no_list:
                    eth_no_list = get_key_no_dictonly(None, keys_list, eth_dict_secondary, True)
                    if eth_no_list: 
                        print(f"eth_dict_istock_secondary found for {eth_no_list}")
                    elif "descent" in keys_list:
                        print(f"descent in keys_list {keys_list}")



            # STORE THE DATA
            print("connecting to DB")

            try:
                with engine.connect() as conn:
                    select_stmt = select([Images]).where(
                        (Images.site_name_id == image_row['site_name_id']) &
                        (Images.site_image_id == image_row['site_image_id'])
                    )
                    row = conn.execute(select_stmt).fetchone()

                    if row is None:
                        insert_stmt = insert(Images).values(image_row)
                        result = execute_query_with_retry(conn, insert_stmt)  # Retry on OperationalError

                        if key_nos_list and result.lastrowid:
                            keyrows = [{'image_id': result.lastrowid, 'keyword_id': keyword_id} for keyword_id in key_nos_list]
                            with engine.connect() as conn:
                                imageskeywords_insert_stmt = insert(ImagesKeywords).values(keyrows)
                                imageskeywords_insert_stmt = imageskeywords_insert_stmt.on_duplicate_key_update(
                                    keyword_id=imageskeywords_insert_stmt.inserted.keyword_id
                                )
                                execute_query_with_retry(conn, imageskeywords_insert_stmt)  # Retry on OperationalError

                        if eth_no_list and result.lastrowid:
                            ethrows = [{'image_id': result.lastrowid, 'ethnicity_id': ethnicity_id} for ethnicity_id in eth_no_list if ethnicity_id is not None]
                            if ethrows:
                                with engine.connect() as conn:
                                    imagesethnicity_insert_stmt = insert(ImagesEthnicity).values(ethrows)
                                    imagesethnicity_insert_stmt = imagesethnicity_insert_stmt.on_duplicate_key_update(
                                        ethnicity_id=imagesethnicity_insert_stmt.inserted.ethnicity_id
                                    )
                                    execute_query_with_retry(conn, imagesethnicity_insert_stmt)  # Retry on OperationalError

                        print("last_inserted_id:", result.lastrowid)
                    else:
                        print('Row already exists:', ind)

            except Exception as e:
                print(f"An error occurred while connecting to DB: {e}")

            finally:
                # Close the session
                session.close()

            # with engine.connect() as conn:
            #     select_stmt = select([Images]).where(
            #         (Images.site_name_id == image_row['site_name_id']) &
            #         (Images.site_image_id == image_row['site_image_id'])
            #     )
            #     row = conn.execute(select_stmt).fetchone()
                
            #     if row is None:
            #         insert_stmt = insert(Images).values(image_row)
            #         result = conn.execute(insert_stmt)
            #         last_inserted_id = result.lastrowid

            #         if key_nos_list and last_inserted_id:
            #             keyrows = [{'image_id': last_inserted_id, 'keyword_id': keyword_id} for keyword_id in key_nos_list]
            #             with engine.connect() as conn:
            #                 imageskeywords_insert_stmt = insert(ImagesKeywords).values(keyrows)
            #                 imageskeywords_insert_stmt = imageskeywords_insert_stmt.on_duplicate_key_update(
            #                     keyword_id=imageskeywords_insert_stmt.inserted.keyword_id
            #                 )
            #                 conn.execute(imageskeywords_insert_stmt)
                    
            #         if eth_no_list and last_inserted_id:
            #             ethrows = [{'image_id': last_inserted_id, 'ethnicity_id': ethnicity_id} for ethnicity_id in eth_no_list if ethnicity_id is not None]
            #             if ethrows:
            #                 with engine.connect() as conn:
            #                     imagesethnicity_insert_stmt = insert(ImagesEthnicity).values(ethrows)
            #                     imagesethnicity_insert_stmt = imagesethnicity_insert_stmt.on_duplicate_key_update(
            #                         ethnicity_id=imagesethnicity_insert_stmt.inserted.ethnicity_id
            #                     )
            #                     conn.execute(imagesethnicity_insert_stmt)
                    
            #         print("last_inserted_id:", last_inserted_id)
            #     else:
            #         print('Row already exists:', ind)
            



            if counter % 1000 == 0:
                save_counter = [counter]
                write_csv(CSV_COUNTOUT_PATH, save_counter)
            
            counter += 1
            ind += 1

'''
# I don't think this is doing anything right now, and I don't know how it is different from ingest_csv
def update_csv():

    # change this for each site ingested
    column_keys = 5
    separator_keys = " "
    # column_site = 8
    column_eth = None
    search_desc_for_keys = True


    with open(CSV_IN_PATH) as file_obj:
        reader_obj = csv.reader(file_obj)
        next(reader_obj)  # Skip header row
        start_counter = get_counter()
        counter = 0
        ind = 0
        
        for row in reader_obj:
            # print(row[1])
            
            if counter < start_counter:
                counter += 1
                continue
            if counter >4001000:
                quit()
            # splitting keys. try is for getty, except is currently set for pexels.
            try:
                keys_list = row[column_keys].lower().split(separator_keys)
            except IndexError:
                print("keys failed")
            # print(keys_list)

            image_row = structure_row_123_asrow(row, ind, keys_list)
            key_nos_list = []
            
            for key in keys_list:
                key_no = unlock_key_plurals_etc(image_row['site_image_id'].lower(), key, keys_dict)
                # print(key_no)
                if key_no:
                    key_nos_list.append(key_no)
            
            # print(key_nos_list)

            if search_desc_for_keys == True:
                desc_key_nos_list = description_to_keys(image_row['description'], image_row['site_image_id'])
                key_nos_list = set(key_nos_list + desc_key_nos_list)
            
            if column_eth:
                # print(key_nos_list)
                eth_no_list = get_eth(row[column_eth].lower(), keys_list)
                print("eth_no_list " , eth_no_list)
            else:
                eth_no_list = None


            # Define the maximum number of retries and the delay between retries
            max_retries = 3
            retry_delay = 20  # in seconds

            # Retry loop
            for retry in range(max_retries):
                try:
                    with engine.connect() as conn:
                        select_stmt = select([images_table]).where(
                            (images_table.c.site_name_id == image_row['site_name_id']) &
                            (images_table.c.site_image_id == image_row['site_image_id'])
                        )
                        row = conn.execute(select_stmt).fetchone()
                        
                        if row is None:

                            print("will insert")
                            # insert_stmt = insert(images_table).values(image_row)
                            # result = conn.execute(insert_stmt)
                            # last_inserted_id = result.lastrowid

                            # if key_nos_list and last_inserted_id:
                            #     keyrows = [{'image_id': last_inserted_id, 'keyword_id': keyword_id} for keyword_id in key_nos_list]
                            #     with engine.connect() as conn:
                            #         imageskeywords_insert_stmt = insert(imageskeywords_table).values(keyrows)
                            #         imageskeywords_insert_stmt = imageskeywords_insert_stmt.on_duplicate_key_update(
                            #             keyword_id=imageskeywords_insert_stmt.inserted.keyword_id
                            #         )
                            #         conn.execute(imageskeywords_insert_stmt)
                            
                            # if eth_no_list and last_inserted_id:
                            #     ethrows = [{'image_id': last_inserted_id, 'ethnicity_id': ethnicity_id} for ethnicity_id in eth_no_list if ethnicity_id is not None]
                            #     if ethrows:
                            #         with engine.connect() as conn:
                            #             imagesethnicity_insert_stmt = insert(imagesethnicity_table).values(ethrows)
                            #             imagesethnicity_insert_stmt = imagesethnicity_insert_stmt.on_duplicate_key_update(
                            #                 ethnicity_id=imagesethnicity_insert_stmt.inserted.ethnicity_id
                            #             )
                            #             conn.execute(imagesethnicity_insert_stmt)
                            
                            # print("last_inserted_id:", last_inserted_id)
                        else:
                            print('Row already exists:', ind)
            
                        break  # If the execution reaches here without exceptions, exit the loop

                except OperationalError as e:
                    print(f"Database connection error: {e}")

                    # Retry after a delay
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue

            else:
                # If the loop completes without a successful connection, handle the error
                print(f"Failed to connect to the database after {max_retries} attempts.")

            if counter % 1000 == 0:
                save_counter = [counter]
                write_csv(CSV_COUNTOUT_PATH, save_counter)
            
            counter += 1
            ind += 1


    # print("inserted")
'''

if __name__ == '__main__':
    print(sig)
    try:
        init_csv(CSV_NOKEYS_PATH,IMG_KEYWORD_HEADERS)
        init_csv(CSV_IMAGEKEYS_PATH,IMG_KEYWORD_HEADERS)
        keys_dict = make_key_dict(KEYWORD_PATH)
        print("this many keys", len(keys_dict))
        locations_dict = make_key_dict(LOCATION_PATH)
        locations_dict_alt = make_key_dict_col3(LOCATION_PATH)
        locations_dict_AA = make_key_dict_col7(LOCATION_PATH)
        print(locations_dict_AA)
        print("this many locations", len(locations_dict))

        ingest_csv()
    except KeyboardInterrupt as _:
        print('[-] User cancelled.\n', flush=True)
    except Exception as e:
        print('[__main__] %s' % str(e), flush=True)
        print(traceback.format_exc())
        # or
        print(sys.exc_info()[2])





