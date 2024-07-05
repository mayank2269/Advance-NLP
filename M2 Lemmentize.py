import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

text="playing with the nltk liberary ,having fun"
token=word_tokenize(text)

# Apply WordNetLemmatizer


lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("plays", 'v'))
print(lemmatizer.lemmatize("played", 'v'))
print(lemmatizer.lemmatize("play", 'v'))
print(lemmatizer.lemmatize("playing", 'v'))

# Lemmatize without POS tagging
lemmatized_words = [lemmatizer.lemmatize(word) for word in token]
print("Lemmatized Words (without POS):", lemmatized_words)


# Function to get POS tag for lemmatization
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Lemmatize with POS tagging
lemmatized_words_pos = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in token]
print("Lemmatized Words (with POS):", lemmatized_words_pos)
