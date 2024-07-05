import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np

corpus = [
    "I love machine learning",
    "Machine learning is fun"
]
# Tokenize
token = [word_tokenize(sentence.lower()) for sentence in corpus]
print("Tokenized Corpus:", token)

# Flatten the list and create a vocabulary
vocab = list(set(word for sentence in token for word in sentence))
print("Vocabulary:", vocab)

# create a Bag of Words vector
def create_bow_vector(sentence, vocabulary):
    word_count = Counter(sentence)
    return np.array([word_count[word] for word in vocabulary])

# Create BoW vectors for each sentence in the tokenized corpus
bow_vectors = np.array([create_bow_vector(sentence, vocab) for sentence in token])
print("Bag of Words Vectors:\n", bow_vectors)