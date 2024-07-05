import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

corpus=[
     "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Tokenize
stop_words = set(stopwords.words('english'))
def tokenize(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]  # Remove tokens that are not alphabetic
    tokens = [t for t in tokens if t not in stop_words]  # Remove stopwords
    return tokens

# Tokenize and preprocess each document in the corpus
tokenized_corpus = [tokenize(doc) for doc in corpus]

# tfidf vectorisation 
# Convert tokenized corpus into strings (required by TfidfVectorizer)
preprocessed_corpus = [' '.join(tokens) for tokens in tokenized_corpus]
# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
# Fit and transform the corpus to TF-IDF vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_corpus)
# Get feature names (terms)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Print TF-IDF matrix (sparse format)
print("TF-IDF Matrix (sparse format):\n", tfidf_matrix)


# Convert sparse matrix to dense matrix (for better understanding)
dense_tfidf_matrix = tfidf_matrix.toarray()
print("\nTF-IDF Matrix (dense format):\n", dense_tfidf_matrix)


# Print feature names
print("\nFeature Names (Terms):\n", feature_names)