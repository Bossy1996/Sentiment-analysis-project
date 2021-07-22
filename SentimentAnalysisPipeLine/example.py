import nltk
from nltk.classify.decisiontree import f

nltk.download([
    "names",
    "stopwords",
    "state_union",
    "twitter_samples",
    "movie_reviews",
    "averaged_perceptron_tagger",
    "vader_lexicon",
    "punkt",
])

w = nltk.corpus.shakespeare.words()

words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]

stopwords = nltk.corpus.stopwords.words("english")

words = [w for w in words if w.lower() not in stopwords]

from pprint import pp, pprint

text = """
For some quick analisys, creating a corpus could be overkill.
If all you need is a word list,
there are simpler ways to achieve that goal.
"""

pprint(nltk.word_tokenize(text), width=79, compact=True)

words: list[str] = nltk.word_tokenize(text)

fd = nltk.FreqDist(words)

fd.most_common(3)
fd.tabulate(3)

fd["America"]
fd["america"]
fd["AMERICA"]

lower = nltk.FreqDist([w.lower() for w in fd])

