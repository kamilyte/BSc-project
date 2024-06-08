import sys
sys.path.append('/Users/kamile/Desktop/Bachelor-Project/BSc-project')
import writing_quality_v2.clean_text as clean_text
from paper import text
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import string

def extract_bigrams(text):
    words = clean_text.preprocess_text(text)
    bigram_finder = BigramCollocationFinder.from_words(words)
    scored_bigrams = bigram_finder.score_ngrams(BigramAssocMeasures.likelihood_ratio)
    return scored_bigrams

def calculate_ads(bigrams):
    if not bigrams:
        return 0
    total_score = sum(score for bigram, score in bigrams)
    m = len(bigrams)
    ads = total_score / m
    return ads

def calculate_adsn(ads, word_count):
    if word_count == 0:
        return 0
    adsn = ads / word_count
    return adsn


bigrams = extract_bigrams(text)
#print(bigrams)
print(calculate_ads(bigrams))
word_count = len(clean_text.preprocess_text(text))
print(calculate_adsn(calculate_ads(bigrams), word_count))
