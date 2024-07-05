import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Load stopwords
stop_words = set(stopwords.words('english'))

text="playing with the nltk liberary ,having fun"
token=word_tokenize(text)

gotwords= [w for w in token if w.lower() not in stop_words]
print(gotwords)

# for Custom stopwords
# custom_stop_words = set(stopwords.words('english')).union({'example', 'text'})
