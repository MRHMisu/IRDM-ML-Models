import re
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()


def count_term_frequency(passage):
    term_frequency = {}
    tokens = preprocess_text(passage)
    for token in tokens:
        term_frequency[token] = tokens.count(token)
    return term_frequency


def preprocess_text(passage):
    converting_lowercase = to_lowercase(passage)
    after_removing_special_characters = remove_specialCharacters(converting_lowercase)
    after_removing_numbers = remove_numbers(after_removing_special_characters);
    after_stop_word = remove_stopwords(after_removing_numbers);
    return after_stop_word


def to_lowercase(token):
    return token.lower()


def remove_specialCharacters(token):
    translator = str.maketrans('', '', string.punctuation)
    return token.translate(translator)


def remove_numbers(token):
    result = re.sub(r'\d+', '', token)
    return result


def remove_stopwords(token):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(token)
    filtered_text = [remove_whitespace(word) for word in word_tokens if word not in stop_words]
    return filtered_text


def remove_whitespace(token):
    return " ".join(token.split())


def stem_words(token):
    word_tokens = word_tokenize(token)
    stems = [stemmer.stem(word) for word in word_tokens]
    return stems
