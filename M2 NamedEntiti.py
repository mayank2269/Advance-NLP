from nltk.tokenize import word_tokenize
import nltk
nltk.download('words')
text="playing with the nltk liberary ,having fun"
token=word_tokenize(text)
pos_tags = nltk.pos_tag(token)
print("POS Tags:", pos_tags)

# Apply NER
named_entities = nltk.ne_chunk(pos_tags)
print("Named Entities:", named_entities)
# Visualize named entities
named_entities.draw()
