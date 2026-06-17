import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

SYNONYMS = {
    "fees": "tuition",
    "fee": "tuition",
    "cost": "tuition",
    "price": "tuition",
    "medical": "clinic",
    "health": "clinic",
    "contact": "email phone",
    "number": "phone",
    "apply": "admission",
    "application": "admission",
    "dorm": "accommodation",
    "hostel": "accommodation",
    "room": "accommodation"
}

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def preprocess(text):
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()

    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = word_tokenize(text)

    tokens = [t for t in tokens if t not in STOP_WORDS]

    # Synonym expansion
    expanded = []
    for t in tokens:
        if t in SYNONYMS:
            expanded.extend(SYNONYMS[t].split())
        else:
            expanded.append(t)
    tokens = expanded

    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)
