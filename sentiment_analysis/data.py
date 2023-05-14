import os
import re
import json
import pickle
import emot
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from config import config

class LabelEncoder(object):
    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index or {}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
    
    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y):
        classes = np.unique(y)
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self
    
    def encode(self, y):
        encoded = np.zeros((len(y)), dtype=int)
        for i, item in enumerate(y):
            encoded[i] = self.class_to_index[item]
        return encoded
    
    def decode(self, y):
        classes = []
        for _, item in enumerate(y):
            classes.append(self.index_to_class[item])
        return classes

    def save(self, filepath):
        with open(filepath, "w") as fp:
            contents = {"class_to_index": self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "r") as fp:
            kwargs = json.load(fp)
            return cls(**kwargs)


# NLTK's default stopwords
tokenizer = TweetTokenizer()
stemmer = PorterStemmer()
EMOTICONS = emot.emo_unicode.EMOTICONS_EMO

with open(config.EMOJI_DICT_FILE, "rb") as fp:
    emoji_dict = pickle.load(fp)

def preprocess(df, strip_stopwords=True, strip_links=True, strip_emojis=False,
            strip_punctuations=True, expand_shortcuts=True, stem=True, lower=True):
    df["content"] = df["content"].apply(
        clean_text,
        strip_stopwords=strip_stopwords,
        strip_links=strip_links,
        strip_emojis=strip_emojis,
        strip_punctuations=strip_punctuations,
        expand_shortcuts=expand_shortcuts,
        stem=stem,
        lower=lower
    )

    df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "text_emotion_processed.csv"), index=False)

    return df

def get_data_splits(X, y, train_size=0.7):
    """Generate balanced data splits."""
    X_train, X_, y_train, y_ = train_test_split(
        X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_, y_, train_size=0.5, stratify=y_)
    return X_train, X_val, X_test, y_train, y_val, y_test

def clean_text(text: str, **kwargs):
    if kwargs["lower"]:
        text = text.lower()
    if kwargs["strip_links"]:
        text = remove_links(text)
    if kwargs["strip_emojis"]:
        text = convert_emojis(text)
        text = convert_emoticons(text)
    if kwargs["expand_shortcuts"]:
        text = expand_contractions(text)
    if kwargs["strip_stopwords"]:
        text = remove_stopwords(text)
    if kwargs["strip_punctuations"]:
        text = remove_punctuations(text)
    if kwargs["stem"]:
        text = " ".join([stemmer.stem(token) for token in tokenizer.tokenize(text)])
    return remove_extra_spaces(text)


def remove_links(text: str):
    return re.sub(r'http\S+', "", text)

def convert_emojis(text):
    for emot in emoji_dict:
        text = re.sub(r"("+emot+")", "_".join(emoji_dict[emot].replace(",","").replace(":","").split()), text)
    return text

# Function for converting emoticons into word
def convert_emoticons(text):
    for k, v in EMOTICONS.items():
        text = re.sub(u"("+re.escape(k)+")", "_".join(v.replace(",","").split()), text)
    return text

def expand_contractions(text):
    # Define a dictionary of contractions and their expanded forms
    contractions_dict = {
        "aren't": "are not",
        "can't": "cannot",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "how'd": "how did",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "let's": "let us",
        "might've": "might have",
        "must've": "must have",
        "mustn't": "must not",
        "needn't": "need not",
        "ought to": "ought to",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "won't": "will not",
        "would've": "would have",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have",
        "2day": "today",
        "b4": "before",
        "1st": "first",
        "2nd": "second",
        "3th": "third",
        "4th": "fourth",
        "5th": "fifth",
        "6th": "sixth",
        "7th": "seventh",
        "8th": "eighth",
        "9th": "ninth",
        "10th": "tenth"
    }
    
    # Define a regular expression pattern to match contractions
    contraction_pattern = re.compile(r'\b(' + "|".join(contractions_dict.keys()) + r")\b")
    
    # Replace all matches of the contraction pattern with their expanded forms
    text = contraction_pattern.sub(lambda match: contractions_dict[match.group(0)], text)
    
    return text

def remove_punctuations(text):
    tokens = tokenizer.tokenize(text)

    # Define a regular expression pattern to match all punctuation except "@" and "#"
    punctuation_pattern = re.compile(r'[^\w\s@#\!]')

    # Replace all matches of the punctuation pattern with an empty string
    tokens = [punctuation_pattern.sub("", token) for token in tokens]
    
    # Join tokens
    text = " ".join(tokens)
    # Return the updated text
    return text

def remove_extra_spaces(text):
    text = text.strip()
    text = " ".join(text.split())
    return text

def remove_stopwords(text):
    pattern = re.compile(r'\b(' + r"|".join(config.STOPWORDS) + r")\b\s*")
    text = pattern.sub('', text)
    return text