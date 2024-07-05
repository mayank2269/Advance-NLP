import numpy as np
import nltk
from nltk.tokenize import word_tokenize

# Sample text
text = "cat dog fish cat dog"
words = word_tokenize(text)
print("Tokenized Words:", words)

# Create vocabulary
vocabulary = list(set(words))
print("Vocabulary:", vocabulary)

def one_hot_encode(word, vocabulary):
    # Initialize a zero vector
    vector = np.zeros(len(vocabulary))
    # Find the index of the word in the vocabulary
    index = vocabulary.index(word)
    # Set the corresponding position to 1
    vector[index] = 1
    return vector

# One-Hot Encode each word
one_hot_vectors = np.array([one_hot_encode(word, vocabulary) for word in words])
print("One-Hot Encoded Vectors:\n", one_hot_vectors)