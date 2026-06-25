import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

for _resource in ("punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    nltk.download(_resource, quiet=True)

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
    "room": "accommodation",
    "office": "department",
    "dept": "department",
    "alumni": "graduate"
}

STOP_WORDS = set(stopwords.words("english"))

from nltk.stem import PorterStemmer
_STEMMER = PorterStemmer()


def preprocess(text):
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS]

    expanded = []
    for t in tokens:
        if t in SYNONYMS:
            expanded.extend(SYNONYMS[t].split())
        else:
            expanded.append(t)
    tokens = expanded

    tokens = [_STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)