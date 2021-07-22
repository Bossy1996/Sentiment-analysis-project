import nltk

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