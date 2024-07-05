import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import nltk


# data preprationnnnnnnnn


corpus = [
    "The quick brown fox jumps over the lazy dog",
    "Never jump over the lazy dog quickly",
    "A quick brown dog outpaces a quick fox"
]

# Tokenize 
processed_corpus = [simple_preprocess(doc) for doc in corpus]
# Remove stopwords
stop_words = stopwords.words('english')
processed_corpus = [[word for word in doc if word not in stop_words] for doc in processed_corpus]

print(processed_corpus)



# train modellllllll
model = Word2Vec(
    sentences=processed_corpus,
    vector_size=100,  # Dimension of the word vectors
    window=5,         # Maximum distance between the current and predicted word within a sentence
    min_count=1,      # Ignores all words with total frequency lower than this
    workers=4,        # Number of worker threads to train the model
    sg=0              # CBOW if 0, Skip-gram if 1
)

# Save the model
model.save("word2vec.model")



# load modelllllllllllllllll
model = Word2Vec.load("word2vec.model")

# use modelllllllllll
# vectorrrr for a wordd
word_vector = model.wv['fox']
print(word_vector)

# similar wordd
similar_words = model.wv.most_similar('fox')
print(similar_words)

# simialarity between wordss
similarity = model.wv.similarity('fox', 'dog')
print(similarity)

# odd one outt
odd_one_out = model.wv.doesnt_match(['quick', 'brown', 'jumps', 'fox'])
print(odd_one_out)





# -----------------------------VISUALISATION WALA--------------------------------------
import gensim
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# load model above wala
model = Word2Vec.load("word2vec.model")

# Extract word vectors
words = list(model.wv.index_to_key)  # List of words in the model's vocabulary
word_vectors = model.wv[words]       # Corresponding word vectors


# dim reduction
pca = PCA(n_components=2)
word_vectors_2d = pca.fit_transform(word_vectors)

# plot 
plt.figure(figsize=(12, 8))

plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1])

for i, word in enumerate(words):
    plt.annotate(word, xy=(word_vectors_2d[i, 0], word_vectors_2d[i, 1]))

plt.title("Word2Vec Word Vectors Visualized with PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()



