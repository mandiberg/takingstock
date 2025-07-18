"""
make_wordcloud_pdf.py
Tench Cholnoky · macOS Sequoia 15.5 · Python 3.10
"""

import os, tempfile, math
import pandas as pd
import random
import numpy as np
import csv
from wordcloud import WordCloud                          
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


import gensim
from gensim import corpora
from nltk.stem import WordNetLemmatizer, SnowballStemmer

# ---------- CONFIG -----------------------------------------------------------
INPUT_PATH  = "../input_csvs/word_cloud/"
STOPWORD_PATH = "../model_files/"
MODEL_PATH = '../model_files/'
OUTPUT_PATH = "../outputs/word_cloud/"

PDF_DATA = {}
OUT_PDF     = "../outputs/word_cloud/wordcloud"  # final file
FONT_FILE   = "../fonts/CrimsonText-Regular.ttf"
FONT_NAME   = "CrimsonText"
PAGE_SIZE   = [432, 648]  

#batch Processing
BATCH_PROCESS = False
PROCESS_SELECT = [27]
CSV_LIST = {}

#cutoff for how many rows of the CSV to add to the textcloud
CUTOFF = True
NUM_ROWS = 300
print("CUTOFF is on")
print("NUM_ROWS is", NUM_ROWS)

EXPORT_EVERY_STEP = False
if EXPORT_EVERY_STEP:
    print("EXPORT_EVERY_STEP is on")


i = 0

# Word-cloud cosmetics
FONT_MIN = 1         # adjust to taste
WC_WIDTH, WC_HEIGHT = 3200, 4800    # px; higher = sharper
BACKGROUND = "white"      
WEIGHT_MIN, WEIGHT_MAX = 0, .08
# -----------------------------------------------------------------------------
def analyze_csv(input_csv, input_path, num_rows):
    df = pd.read_csv(input_path+input_csv)
    df = df.dropna()
    if CUTOFF:
        df = df.head(num_rows)
    return input_csv, df, 


def get_document_topic_weights_simple(model, bow_vector, topic_id):
    # Handle invalid/empty cases
    if not bow_vector:
        return 0
    
    # Check if it's a list containing 'none' or 'blank'
    if isinstance(bow_vector, list) and len(bow_vector) > 0:
        first_item = bow_vector[0]
        if first_item == "none":
            return "none"
        elif first_item == "blank":
            return "blank"
    
    # bow_vector should be a list of (word_id, count) tuples like [(0, 1), (5, 2)]
    doc_topics = model.get_document_topics(bow_vector, minimum_probability=0)
    for w in doc_topics:
        if w[1] > .1:
            print("topic match, weight:", w[1], "topic", w[0])
    
    topic_weight = 0
    for topic, weight in doc_topics:
        if topic == topic_id:
            topic_weight = weight
            break
    
    return topic_weight

def map_values_to_range(input_list):
    numerical_values = []
    result = []
    
    for value in input_list:
        if value == "none":
            result.append("outline")
        elif value == "blank":
            result.append("italic")
        else:
            numerical_values.append(float(value))
            result.append(None)
    
    # Apply log transform to spread out values
    log_values = np.log1p(numerical_values)  # log1p handles zero values safely
    
    min_val = np.min(log_values)
    max_val = np.max(log_values)
    
    if min_val == max_val:
        mapped_values = [1.0] * len(log_values)
    else:
        mapped_values = [(max_val - val) / (max_val - min_val) for val in log_values]
    
    mapped_index = 0
    for i, value in enumerate(input_list):
        if value != "none" and value != "blank":
            result[i] = mapped_values[mapped_index]
            mapped_index += 1
    return result
# ---------- 0) BATCH  -----------------------------------------------------
if BATCH_PROCESS:
    print("Batch processing")
    for file in sorted(os.listdir(INPUT_PATH)):
        if file.endswith(".csv"):
            csv_info = analyze_csv(file, INPUT_PATH, NUM_ROWS)
            CSV_LIST.update({csv_info[0] : csv_info[1]})  # Use topic number (csv_info[0]) as key
#If not batch, add only the csvs selected from the list
elif len(PROCESS_SELECT) > 0:
    print("processing CSVS " + str(PROCESS_SELECT))
    for file in sorted(os.listdir(INPUT_PATH)):
        for i in PROCESS_SELECT:
            if str(i) in file:
                 if file.endswith(".csv"):
                    csv_info = analyze_csv(file, INPUT_PATH, NUM_ROWS)
                    CSV_LIST.update({csv_info[0] : csv_info[1]})  # Use topic number (csv_info[0]) as key
else:
    print("Add numbers to PROCESS_SELECT or turn on batch processing")

#export every step(0)
if EXPORT_EVERY_STEP:
    print("Export every step(0), preprocessed csvs")
    print("exporting CSV_LIST, ", len(CSV_LIST), " csv(s) to export")
    i = 0
    for csv in CSV_LIST:
        CSV_LIST[csv].to_csv(OUTPUT_PATH+str(i)+"_EXPORT_STEP_0_"+csv)
        i+=1

#---------- 0.5) Preprocessing and load model  -----------------------------------------------------
def preprocess(text, MY_STOPWORDS):
    result = []
    text = text.lower()
    for token in gensim.utils.simple_preprocess(text):
        if token not in MY_STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


stemmer = SnowballStemmer('english')
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# open and read a csv file, and assign each row as an element in a list
def read_csv(file_path):
    with open(file_path, 'r') as file:
        data = file.read().replace('\n', '')
    return data


#CUSTOM GRAYSCALE COLOUR FUNCTION 
# def gray_color(word, font_size, position, orientation, random_state=None, **kw):
    # """Return an rgb() string whose gray level comes from the CSV's 'relation'."""
    # rel = relations.get(word, 0)
    # if rel == "outline":
    #     return f"rgb(1, 0, 0)"
    # elif rel == "italic":
    #     return f"rgb(0, 1, 0)"
    # g   = int( (1 - rel) * 255 )           # 0: black → 255: white
    # return f"rgb({g}, {g}, {g})"

#load the model
lda_model_tfidf = gensim.models.LdaModel.load(MODEL_PATH+'model')
lda_dict = corpora.Dictionary.load(MODEL_PATH+'model.id2word')

# ---------- 1) LOAD DATA -----------------------------------------------------
GENDER_LIST = read_csv(os.path.join(STOPWORD_PATH, "stopwords_gender.csv"))
ETH_LIST = read_csv(os.path.join(STOPWORD_PATH, "stopwords_ethnicity.csv"))
AGE_LIST = read_csv(os.path.join(STOPWORD_PATH, "stopwords_age.csv"))                       
SKIP_TOKEN_LIST = read_csv(os.path.join(STOPWORD_PATH, "skip_tokens.csv"))   
MY_STOPWORDS = gensim.parsing.preprocessing.STOPWORDS.union(set(GENDER_LIST+ETH_LIST+AGE_LIST)) 

#set up dictionary
DICT_PATH=os.path.join(MODEL_PATH,"dictionary.dict")
dictionary = corpora.Dictionary.load(MODEL_PATH+'model.id2word')

# Calculate min/max once outside the function
# valid_scores = [v for v in key_score_dict.values() if v is not None]
# MIN_SCORE = min(valid_scores)
# MAX_SCORE = max(valid_scores)
MIN_SCORE = 0
MAX_SCORE = .1

def gray_color(word, font_size, position, orientation, random_state=None, **kw):
    """Return an rgb() string whose gray level comes from the key_score_dict."""
    score = key_score_dict.get(word, None)
    print(word, score)
    if score is None:
        return f"rgb(255, 255, 255)"  # Medium gray for None values
    
    # Normalize score to 0-1 range
    normalized_score = (score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)
    
    # Convert to grayscale (0: black → 255: white)
    g = int(normalized_score * 255)
    return f"rgb({g}, {g}, {g})"

def get_all_topics_words(lda_model):
    # Get all topics with their word probabilities
    topics = lda_model.print_topics(num_topics=-1, num_words=len(lda_model.id2word))

    # Or get the topic-word matrix
    topic_word_matrix = lda_model.get_topics()
    # print(topic_word_matrix.shape)  # (num_topics, vocab_size)

    # Get word probabilities for a specific topic
    # topic_0_words = lda_model.show_topic(0, topn=len(lda_model.id2word))

    # Get all words for all topics
    all_topics_words = []
    for topic_id in range(lda_model.num_topics):
        topic_words = lda_model.show_topic(topic_id, topn=len(lda_model.id2word))
        # print(topic_id, topic_words)
        all_topics_words.append(topic_words)
    return all_topics_words


all_topics_words = get_all_topics_words(lda_model_tfidf)
#process each csv
for csv in CSV_LIST:
    CSV_NUMBER = csv.split('topic_')[1].split('_counts.csv')[0]
    print("Processing: " + CSV_NUMBER)
    this_topics_words = dict(all_topics_words[int(CSV_NUMBER)])
   

    df = pd.read_csv(INPUT_PATH+csv).dropna(subset=["description", "count"])

    keyword_list = []
    remove_empty_list = []
    unproccessed_word_dict = {}
    for row in df.iterrows():
        #preprocess the text
        unproccessed_word = row[1][1]
        tokens = preprocess(unproccessed_word, MY_STOPWORDS)
        print(tokens)
        keyword_list.append(tokens)
        
        # Check if tokens is empty
        if not tokens:  # This is True when tokens is an empty list
            unproccessed_word_dict["FOOBAR"] = unproccessed_word
        else:
            print('Unprocessed word dict ', unproccessed_word_dict.get(tokens[0], 'Key not found'))
            unproccessed_word_dict[tokens[0]] = unproccessed_word
    #remove empty lists
    for each in keyword_list:
        if each == []:
            remove_empty_list.append(each)
    for each in remove_empty_list:
        keyword_list.remove(each)

    if CUTOFF and len(keyword_list) > NUM_ROWS:
        keyword_list = keyword_list[:NUM_ROWS]

    #export every step(1)
    if EXPORT_EVERY_STEP:
        print("Export every step(1), keyword list")
        export_every_step_df = pd.DataFrame(keyword_list)
        export_every_step_df.to_csv(OUTPUT_PATH+"EXPORT_STEP_1_"+csv)

    key_score_dict = {}
    for key in keyword_list:

        key_score = this_topics_words.get(key[0], None)
        unproccessed_word = unproccessed_word_dict[key[0]]
        key_score_dict[unproccessed_word] = key_score
    #creating bow vector and tokenizing

    # #if the bow vector is empty (only happens if words are in stopword list), replace with "none" for outlining text later
    # bow_vector = []
    # for each in keyword_list:
    #     bow_vector.append(dictionary.doc2bow(each))
    # for i in range(len(bow_vector)):
    #     if len(bow_vector[i]) == 0:
    #         bow_vector[i] = ["none"]
    #     elif len(bow_vector[i]) > 1:
    #         bow_vector[i] = [bow_vector[i][0]]

#  #export every step(2)
#     if EXPORT_EVERY_STEP:
#         print("Export every step(2), bow vector")
#         export_every_step_df = pd.DataFrame(bow_vector)
#         export_every_step_df.to_csv(OUTPUT_PATH+"EXPORT_STEP_2_"+csv)

    sorted_topics = []

    # #get weights in relation to specific topic currently being used
    # for i in range(len(bow_vector)):         
    #     weights = get_document_topic_weights_simple(lda_model_tfidf, bow_vector[i], int(CSV_NUMBER))  
    #     sorted_topics.append(weights)


    for i in range(len(sorted_topics)):
        if sorted_topics[i]== 0.015625:
            sorted_topics[i] = "blank"


 #export every step(3)
    if EXPORT_EVERY_STEP:
        print("Export every step(3), sorted topics")
        export_every_step_df = pd.DataFrame(sorted_topics)
        export_every_step_df.to_csv(OUTPUT_PATH+"EXPORT_STEP_3_"+csv)



    # Frequencies for WordCloud
    #may need to check
    freqs  = dict(zip(df["description"], df["count"]))
   #relations = dict(zip(df['description'], map_values_to_range(sorted_topics)))
    relations = dict(zip(df['description'], sorted_topics))
    PDF_DATA[CSV_NUMBER] = dict(zip(freqs, relations))

    # ---------- 3) BUILD WORD CLOUD ---------------------------------------------
    wc = (
        WordCloud(width=WC_WIDTH,
                height=WC_HEIGHT,
                background_color=BACKGROUND,
                prefer_horizontal=1.0,
                min_font_size=FONT_MIN,
                color_func=gray_color, ##Interpret as colormap rather than single color, use LLM to process in between steps
                font_path=FONT_FILE)
        .generate_from_frequencies(freqs)
    )

    # Save to a temp PNG
    tmp_png = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    wc.to_file(tmp_png.name)

    # ---------- 4) RENDER WORD CLOUD ON A PDF PAGE ------------------------------
    tmp_wc_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_FILE))

    c = canvas.Canvas(tmp_wc_pdf.name, pagesize=PAGE_SIZE)
    img = ImageReader(tmp_png.name)

    # Fit image to page (keep aspect ratio, centred)
    img_width, img_height = wc.to_image().size
    page_w, page_h = PAGE_SIZE
    scale = min((page_w / img_width) * 0.9, (page_h / img_height) * 0.9)  # 90 % inset
    draw_w, draw_h = img_width * scale, img_height * scale
    x = (page_w - draw_w) / 2
    y = (page_h - draw_h) / 2
    c.drawImage(img, x, y, width=draw_w, height=draw_h, mask="auto")
    c.showPage()
    c.save()

    # ---------- 5) CREATE NEW PDF WITH WORD CLOUD PAGE ONLY ---------------------

    pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_FILE))
    temp_out = OUT_PDF+CSV_NUMBER+'.pdf'
    c = canvas.Canvas(OUT_PDF+CSV_NUMBER+'.pdf', pagesize=PAGE_SIZE)
    img = ImageReader(tmp_png.name)
    print("c:" ,c)
    print("c type", type(c))

    # Fit image to page (keep aspect ratio, centred)
    img_width, img_height = wc.to_image().size
    page_w, page_h = PAGE_SIZE
    scale = min((page_w / img_width) * 0.9, (page_h / img_height) * 0.9)  # 90 % inset
    draw_w, draw_h = img_width * scale, img_height * scale
    x = (page_w - draw_w) / 2
    y = (page_h - draw_h) / 2
    c.drawImage(img, x, y, width=draw_w, height=draw_h, mask="auto")
    c.showPage()
    c.save()

    print(f"✅ Word-cloud PDF created → {temp_out}")

