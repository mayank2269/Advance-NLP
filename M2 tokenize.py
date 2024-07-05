import nltk 
from nltk.tokenize import word_tokenize,sent_tokenize,casual_tokenize
text= "lets start the nlp with a bang . are u guys ready"
print(word_tokenize(text))
print(sent_tokenize(text))
print(casual_tokenize(text))

# tokenise to diff languages
txt="Hola amigo. Estoy bien. Como andas?"
print(word_tokenize(txt,'spanish') )


# remove stop words 
from nltk.corpus import stopwords
# nltk.download('stopwords')     // download the stopwords resourse
ans=word_tokenize(text)
stwords=set(stopwords.words('english')) #importing the stop words in english

new = [w for w in ans if  w.lower() not in stwords] # for lowercase the words 
print(new)
 

# coustom token with regular expression 
import re
custom_tokens = re.findall(r'\b\w+\b', text)
print("Custom Tokens:", custom_tokens)






