# Text preprocessing is usually the first step you’ll take when faced with an NLP task.
# Noise removal, Tokenization, Normalization: Lemmatization, Stemming

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from part_of_speech import get_part_of_speech

text = """So many squids are jumping out of suitcases these days that you can barely go anywhere without seeing one
 burst forth from a tightly packed valise. I went to the dentist the other day, and sure enough
 I saw an angry one jump out of my dentist's bag within minutes of arriving. She hardly even noticed."""

cleaned = re.sub('\W', ' ', text)
tokenized = word_tokenize(cleaned)

stemmer = PorterStemmer()
stemmed = [stemmer.stem(token) for token in tokenized]

# By default lemmatize() treats every word as a noun unless you add the pos argument
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(token, pos=get_part_of_speech(token)) for token in tokenized]
print(lemmatized)