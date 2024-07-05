import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Load stopwords
stop_words = set(stopwords.words('english'))

text="playing with the nltk liberary ,having fun"
token=word_tokenize(text)
# Apply POS tagging
pos_tags = nltk.pos_tag(token)
print("POS Tags:", pos_tags)



# visualise the POS 

from nltk import Tree
from nltk.draw import tree

# Sample text
text = "The quick brown fox jumps over the lazy dog."
words = nltk.word_tokenize(text)
pos_tags = nltk.pos_tag(words)

# Convert POS tags to a Tree
pos_tree = Tree('S', [Tree(tag, [word]) for word, tag in pos_tags])

# Draw the tree
tree.draw_trees(pos_tree)
