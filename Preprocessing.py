from emoji import demojize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string

# All Tweet normaliser code was sourced from https://github.com/VinAIResearch/BERTweet/blob/c595d21749591ca43ddcda66de0facd3a14ec23b/TweetNormalizer.py
tokenizer = TweetTokenizer()

# Replace user tags and urls
def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        # Replace emojis by text description
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token

def normalizeTweet(tweet):
    if(str(tweet)!="nan"):
        tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
        normTweet = " ".join([normalizeToken(token) for token in tokens])
        # Replace acronyms and contractions
        normTweet = (
            normTweet.replace("cannot ", "can not ")
            .replace("n't ", " n't ")
            .replace("n 't ", " n't ")
            .replace("ca n't", "can't")
            .replace("ai n't", "ain't")
        )
        normTweet = (
            normTweet.replace("'m ", " 'm ")
            .replace("'re ", " 're ")
            .replace("'s ", " 's ")
            .replace("'ll ", " 'll ")
            .replace("'d ", " 'd ")
            .replace("'ve ", " 've ")
        )
        normTweet = (
            normTweet.replace(" p . m .", "  p.m.")
            .replace(" p . m ", " p.m ")
            .replace(" a . m .", " a.m.")
            .replace(" a . m ", " a.m ")
        )
        return " ".join(normTweet.split())

#Remove stopwords that might affect the classification of our tweets
# Code from https://towardsdatascience.com/cross-topic-argument-mining-learning-how-to-classify-texts-1d9e5c00c4cc
def remove_stopwords(text):
    if(text is not None and isinstance(text, str)):
        stpword = stopwords.words('english')
        no_punctuation = [char for char in text if char not in
            string.punctuation]
        no_punctuation = ''.join(no_punctuation)
        return ' '.join([word for word in no_punctuation.split() if
            word.lower() not in stpword])

# Preprocess tweets by removing stopwords and normalising
def preprocess(text):
    return remove_stopwords(normalizeTweet(text))