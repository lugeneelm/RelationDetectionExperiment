import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, log_loss
import numpy as np
from SentimentFeatureExtractor import SentimentExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from Preprocessing import remove_stopwords, normalizeTweet

if __name__ == '__main__':
    # Extract dataset from csv file
    df= pd.read_csv("../Datasets/PAIRED-ANNOTATED.csv")

    # Preprocess tweet text
    df["Argument1"]= df["Argument1"].apply(remove_stopwords)
    df["Argument2"]= df["Argument2"].apply(remove_stopwords)

    # Split dataset into training and evaluation, 80-20
    trainingSize=int(np.rint(df.shape[0]*0.8))
    df.dropna(inplace=True)
    train_df= df[:trainingSize]
    eval_df= df[trainingSize:]

    # Create sentiment analysis extractor
    sentiment= SentimentExtractor()

    # Predict sentiments for all dataset arguments
    A1_train_sentiment= train_df["Argument1"].apply(sentiment.predictSentiment).tolist()
    A2_train_sentiment= train_df["Argument2"].apply(sentiment.predictSentiment).tolist()
    A1_eval_sentiment= eval_df["Argument1"].apply(sentiment.predictSentiment).tolist()
    A2_eval_sentiment= eval_df["Argument2"].apply(sentiment.predictSentiment).tolist()

    # Transform predictions into one-hot encodings
    A1_train_sentiment= pd.get_dummies(A1_train_sentiment)
    A2_train_sentiment= pd.get_dummies(A2_train_sentiment)
    A1_eval_sentiment= pd.get_dummies(A1_eval_sentiment)
    A2_eval_sentiment= pd.get_dummies(A2_eval_sentiment)

    # Combine sentiment predictions of tweets in each pair
    Xs_train = pd.concat([A1_train_sentiment,A2_train_sentiment] , axis=1).to_numpy()
    Xs_eval = pd.concat([A1_eval_sentiment,A2_eval_sentiment], axis=1).to_numpy()

    # Preprocess tweets by normalising
    train_df.loc[:,["Argument1"]]= train_df["Argument1"].apply(normalizeTweet)
    train_df.loc[:,["Argument2"]]= train_df["Argument2"].apply(normalizeTweet)
    eval_df.loc[:,["Argument1"]]= eval_df["Argument1"].apply(normalizeTweet)
    eval_df.loc[:,["Argument2"]]= eval_df["Argument2"].apply(normalizeTweet)

    # Create and fit vectoriser to training data
    trainingArguments= train_df["Argument1"].append(train_df["Argument2"])
    vectorizer = CountVectorizer()
    vectorizer.fit(trainingArguments)

    # Vectorise all arguments in dataset
    A1_train= vectorizer.transform(train_df["Argument1"]).toarray()
    A2_train= vectorizer.transform(train_df["Argument2"]).toarray()

    A1_eval= vectorizer.transform(eval_df["Argument1"]).toarray()
    A2_eval= vectorizer.transform(eval_df["Argument2"]).toarray()

    # Combine vectorised tweet in each pair
    Xv_train= np.column_stack((A1_train, A2_train))
    Xv_eval= np.column_stack((A1_eval,A2_eval))

    # Combine sentiment analysis vectors and text vectorisation for each pair to form training and evaluation data
    X_train= np.column_stack((Xs_train, Xv_train))
    X_eval= np.column_stack((Xs_eval, Xv_eval))

    Y_train= train_df["Label"]
    Y_eval= eval_df["Label"]

    # Create LR model and fit to training data
    model = LogisticRegression(solver='liblinear', C=0.45).fit(X_train, Y_train)

    # Evaluate LR model
    eval_predicted= model.predict(X_eval)
    f1= f1_score(Y_eval, eval_predicted)
    acc= accuracy_score(Y_eval, eval_predicted)
    eval_probs_predicted= model.predict_proba(X_eval)[:,1]
    Y_eval= Y_eval.to_numpy(dtype=np.float64)
    loss= log_loss(Y_eval, eval_probs_predicted)

    # Print results of evaluation
    print('F1=', f1, ', Accuracy=', acc, ', Loss=', loss)
    # F1= 0.6666666666666666 , Accuracy= 0.5657894736842105 , Loss= 0.8618667447906169
