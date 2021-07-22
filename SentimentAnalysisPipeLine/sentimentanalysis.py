import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from random import shuffle
from statistics import mean
from nltk.tokenize import word_tokenize
from nltk.tree import sinica_parse

from numpy import negative

# Creates an instance of the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# # Load twitter sample corpus. strings gives you raw tweets as strings
# tweets = [t.replace("://", "//") for t in nltk.corpus.twitter_samples.strings()]

# def is_positive(tweet: str) -> bool:
#     """True if tweet has positive compound sentiment, False otherwise"""
#     return sia.polarity_scores(tweet)["compound"] > 0

# shuffle(tweets)
# """ for tweet in tweets[:10]:
#     print(">", is_positive(tweet), tweet) """

# Movie Review example
positive_review_ids = nltk.corpus.movie_reviews.fileids(categories=["pos"])
negative_review_ids = nltk.corpus.movie_reviews.fileids(categories=["neg"])
all_review_ids = positive_review_ids + negative_review_ids

def is_positive_review(review_id: str) -> bool:
    """True if the average of all sentence compound socres is positive"""
    text = nltk.corpus.movie_reviews.raw(review_id)
    scores = [
        sia.polarity_scores(sentence)["compound"]
        for sentence in nltk.sent_tokenize(text)
    ]
    return mean(scores) > 0

shuffle(all_review_ids)
correct = 0
for review_id in all_review_ids:
    if is_positive_review(review_id):
        if review_id in positive_review_ids:
            correct += 1
        else:
            if review_id in negative_review_ids:
                correct += 1
print(F"{correct / len(all_review_ids):.2%} correct")

unwanted = nltk.corpus.stopwords.words("english")
unwanted.extend([w.lower() for w in nltk.corpus.names.words()])

def skip_unwanted(pos_tuple):
    word, tag = pos_tuple
    if not word.isalpha() or word in unwanted:
        return False
    if tag.startswith("NN"):
        return False
    return True

positive_words = [word for word, tag in filter(skip_unwanted, nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["pos"])))]
negative_words = [word for word, tag in filter(skip_unwanted, nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["neg"])))]

positive_fd = nltk.FreqDist(positive_words)
negative_fd = nltk.FreqDist(negative_words)

common_set = set(positive_fd).intersection(negative_fd)

for word in common_set:
    del positive_fd[word]
    del negative_fd[word]

top_100_positive = {word for word, count in positive_fd.most_common(100)}
top_100_negative = {word for word, count in negative_fd.most_common(100)}
""" 
positive_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
    w for w in nltk.corpus.movie_reviews.words(categories=["pos"])
    if w.isalpha() and w not in unwanted
])
negative_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
    w for w in nltk.corpus.movie_reviews.words(categories=["neg"])
    if w.isalpha() and w not in unwanted
]) """

# Training and using a Classifier

def extract_features(text):
    features = dict()
    wordcount = 0
    compound_scores = list()
    positive_scores = list()

    for sentence in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sentence):
            if word.lower() in top_100_positive:
                wordcount += 1
        compound_scores.append(sia.polarity_scores(sentence)["compound"])
        positive_scores.append(sia.polarity_scores(sentence)["pos"])

    # Adding 1 to the final compound score to always have positive numbers
    # since some classifiers you'll use later dont' work with negative numbers
    features["mean_compound"] = mean(compound_scores) + 1
    features["mean_positives"] = mean(positive_scores)
    features["wordcount"] = wordcount

    return features