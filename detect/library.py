import os
import re
import pandas as pd
import regex

def load_dataset(filepath):
    df = pd.read_csv(filepath)
    df.columns = ['id','message','class']
    return df


def detect_lang_unicode(sentence):
    words = sentence.split()
    for word in words:
        for char in word:
            if ord(char) in range(2304,2432):
                return "hi"
    return "en"

def extract_hindi_text_documents(df):
    lang = []
    texts = df["message"]
    for text in texts:
        language = detect_lang_unicode(str(text))
        if  language == "hi":
            lang.append("hi")
        else:
            lang.append("en")
    df["lang"] = lang
    hin_df = df[df["lang"]=="hi"]
    return hin_df


def merge_hindi_datasets(filepath1,filepath2):
    df1 = load_dataset(filepath1)
    hin_df1 = extract_hindi_text_documents(df1)
    df2 = load_dataset(filepath2)
    hin_df2 = extract_hindi_text_documents(df2)
    hin_df = pd.concat([hin_df1, hin_df2])
    return hin_df

def getStopwords():
    files = ["helperfiles/stopwords.txt","helperfiles/final_stopwords.txt","helperfiles/new_stopwords.txt"]
    stop_words = ["हैं","है"]
    for file in files:
        f = open(file,"r",encoding="utf-8")
        lines = f.readlines()
        for i in lines:
            i = re.sub("[\n]", "" , i)
            stop_words.append(i)
    stopwords = set(stop_words)
    return stopwords

def clean(text,stopwords):
    mod_text_list = remove_stop_words(text.split(),stopwords)
    return " ".join(mod_text_list)

def remove_stop_words(text_list,stopwords):
    mod_text_list = []
    for word in text_list:
        word = word.strip()
        if word not in stopwords and len(word)>=2 and (not word.isnumeric()) :
            mod_text_list.append(word)
    return mod_text_list

def apply_clean(hin_df,key,stopwords):
    messages = list(hin_df[key])
    mod_messages = []
    for msg in messages:
        mod_messages.append(clean(msg,stopwords))
    hin_df[key] = mod_messages

def custom_analyzer(text):
    words = regex.findall( r'\b\w+\b' , text) # extract words of at least 2 letters
    for w in words:
        yield w


def preprocess_text(df):
    # Define regular expressions for patterns to remove
    url_pattern = r"http\S+"
    username_pattern = r"@\w+"
    hashtag_pattern = r"#\w+"
    emoticon_pattern = r"[<>]?[:;=8][\-o\*\']?[D\)\]\(\]/\\OpP]"
    punctuation_pattern = r"[^\w\s]"
    unwanted_pattern = r"[^a-zA-Z0-9\s]"

    # Remove URLs, usernames, hashtags, and emoticons
    df["message"] = df["message"].apply(lambda x: re.sub(
        url_pattern, "", x) if isinstance(x, str) and detect_lang_unicode(x) == "en" else x)
    df["message"] = df["message"].apply(lambda x: re.sub(
        username_pattern, "", x) if isinstance(x, str) and detect_lang_unicode(x) == "en" else x)
    df["message"] = df["message"].apply(lambda x: re.sub(
        hashtag_pattern, "", x) if isinstance(x, str) and detect_lang_unicode(x) == "en" else x)
    df["message"] = df["message"].apply(lambda x: re.sub(
        emoticon_pattern, "", x) if isinstance(x, str) and detect_lang_unicode(x) == "en" else x)

    # Remove punctuation and unwanted characters
    df["message"] = df["message"].apply(lambda x: re.sub(
        punctuation_pattern, "", x) if isinstance(x, str) and detect_lang_unicode(x)=="en" else x)
    # df["message"] = df["message"].apply(lambda x: re.sub(
    #     unwanted_pattern, "", x) if isinstance(x, str) else x)

    # Convert all words to lowercase and remove extra whitespace
    df["message"] = df["message"].apply(
        lambda x: x.lower().strip() if isinstance(x, str) and detect_lang_unicode(x) == "en" else x)
    df["message"] = df["message"].apply(lambda x: re.sub(
        r"\s+", " ", x) if isinstance(x, str) and detect_lang_unicode(x) == "en" else x)

    return df
