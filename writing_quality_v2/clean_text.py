import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')


def replace_unnecessary_chars(text):
    # replace '- ' and newlines with an empty character
    cleaned_text = text.replace('- ', '')
    cleaned_text = text.replace('\n', '')
    return cleaned_text

# preprocessing text
def preprocess_text(text):
    text = replace_unnecessary_chars(text)
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return words

