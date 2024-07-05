import nltk
from nltk.util import ngrams
from collections import Counter
text="I Love Machine Learning"
token= nltk.word_tokenize(text)

bigram=list(ngrams(token,2))
trigram=list(ngrams(token,3))

print(bigram)
print(trigram)
