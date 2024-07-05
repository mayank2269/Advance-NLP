import nltk 
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import numpy as np


# tokenise
corpus=[
"I Love Machine Learning",
"Machine Learning is fun"
]
token=[word_tokenize(sentence.lower())for sentence in corpus]
print(token)



# ngrammm
# Function to generate N-Grams for a given tokenized sentence
def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# Generate Bigrams for each sentence in the tokenized corpus
bigrams_corpus = [generate_ngrams(sentence, 2) for sentence in token]
print("Bigrams Corpus:", bigrams_corpus)

# Flatten the list of bigrams for vocabulary creation
flattened_bigrams = [bigram for sentence in bigrams_corpus for bigram in sentence]
print("Flattened Bigrams:", flattened_bigrams)



# vocabbbb
# Create a vocabulary of unique bigrams
bigram_vocab = list(set(flattened_bigrams))
print("Bigram Vocabulary:", bigram_vocab)



# vectorizeeeee
# Function to create a Bag of Words vector for N-Grams
def create_ngram_bow_vector(ngrams, vocabulary):
    ngram_count = Counter(ngrams)
    return np.array([ngram_count[ngram] for ngram in vocabulary])

# Create BoW vectors for each sentence in the bigrams corpus
ngram_bow_vectors = np.array([create_ngram_bow_vector(sentence, bigram_vocab) for sentence in bigrams_corpus])
print("N-Gram Bag of Words Vectors:\n", ngram_bow_vectors)
