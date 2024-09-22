from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import contractions
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')


stop_words = set(stopwords.words("english"))


def tfidf_vectorizer():
    vectorizer = pickle.load(open("weights/tfidf_vectorizer.pkl", "rb"))
    return vectorizer

# Expanding contractions


def expand_contractions(text):
    return contractions.fix(text)

# Function to clean data


def process_text(text):
    vectorizer = tfidf_vectorizer()
    wl = WordNetLemmatizer()
    soup = BeautifulSoup(text, "html.parser")  # remove html tags
    text = soup.get_text()
    text = expand_contractions(text)  # expand contractions
    emoji_clean = re.compile("["
                             u"\U0001F600-\U0001F64F"  # emoticons
                             u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                             u"\U0001F680-\U0001F6FF"  # transport & map symbols
                             u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                             u"\U00002702-\U000027B0"
                             u"\U000024C2-\U0001F251"
                             "]+", flags=re.UNICODE)
    text = emoji_clean.sub(r'', text)  # remove emojis
    text = re.sub(r'\.(?=\S)', '. ', text)  # add space after full stop
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = "".join([
        word.lower() for word in text if word not in string.punctuation
    ])  # remove punctuation and convert to lowercase
    text = " ".join([
        wl.lemmatize(word) for word in text.split() if word not in stop_words and word.isalpha()
    ])  # remove stop words and lemmatize
    return vectorizer.transform([text])
