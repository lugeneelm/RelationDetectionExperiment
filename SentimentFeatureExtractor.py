from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax

# Code inspired by https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

# Preprocess text following guidelines specified by  "cardiffnlp/twitter-roberta-base-sentiment-latest"
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') & len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Create sentiment analysis extractor class
class SentimentExtractor():
    # Load pretrained transformer based model and tokeniser
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Use pretrained model and classifier to predict sentiment
    def predictSentiment(self, text):
        tokenised= self.tokenizer(preprocess(text), max_length = 500, truncation = True, padding = True, return_tensors='pt')
        output= softmax(self.model(**tokenised)[0][0].detach().numpy())
        label= np.argmax(output)
        return label