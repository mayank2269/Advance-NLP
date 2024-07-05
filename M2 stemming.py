import nltk
from nltk.stem import PorterStemmer,SnowballStemmer,LancasterStemmer
from nltk.tokenize import word_tokenize

text="playing with the nltk liberary ,having fun"
token=word_tokenize(text)

# Apply Porter Stemmer


# portr = [PorterStemmer().stem(word) for word in token]
# print("Porter Stems:", portr)
# # op=['play', 'with', 'the', 'nltk', 'liberari', ',', 'have', 'fun']              ///fast and simple ans
porter = PorterStemmer()
print(porter.stem("play"))
print(porter.stem("playing"))
print(porter.stem("plays"))
print(porter.stem("played"))

# Apply Lancaster Stemmer


# lan = [LancasterStemmer().stem(word) for word in token]
# print("Lancaster Stems:", lan)
# # op=Lancaster Stems: ['play', 'with', 'the', 'nltk', 'lib', ',', 'hav', 'fun']    /// short ans
lancas=LancasterStemmer()
print(lancas.stem("play"))
print(lancas.stem("playing"))
print(lancas.stem("plays"))


# Apply Snowball Stemmer

# snow = [SnowballStemmer('english').stem(word) for word in token]
# print("Snowball Stems:", snow)
# op=['play', 'with', 'the', 'nltk', 'liberari', ',', 'have', 'fun']               ///robust  ans
snow=SnowballStemmer('english')
print(snow.stem("play"))
print(snow.stem("playing"))
print(snow.stem("plays")) 




