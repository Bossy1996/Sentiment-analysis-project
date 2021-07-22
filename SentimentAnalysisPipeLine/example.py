# NLTK examples of usage

import nltk
from nltk.classify.decisiontree import f
from nltk.corpus.reader import lin

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

# Extracting Concordance and Collocations
# Concordance is a collection of word locations along with their context. You can use concordance to find
# 1. How many times a word appears
# 2. Where each occurrence appears
# 3. What words sorround each occurrence

text = nltk.Text(nltk.corpus.state_union.words())
# .concordance will only print its result into the screen
text.concordance("america", lines=5)

# .concordance_list will give you a usable list of ConcordanceLine objects
concordance_list = text.concordance_list("america", lines=2)
for entry in concordance_list:
    print(entry.line)

words: list[str] = nltk.word_tokenize("""
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
""")

text = nltk.Text(words)
fd = text.vocab() # Equivalent to fd = nltk.FreqDist(words)
fd.tabulate(3)

# Collocations can be made up of two or more words. NLTK provide classes to hanlde several types of collocations:
# Bigrams: Frequent two-word combinations
# Trigrams: Frequent three-word combinations
# Quadgrams: Frequent four-word combinations

words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]
# TrigramCollocationFinder will search for trigrams
finder = nltk.collocations.TrigramCollocationFinder.from_words(words)