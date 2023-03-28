import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, log_loss
import numpy as np
from SentimentFeatureExtractor import SentimentExtractor
from sklearn.linear_model import LogisticRegression
from Preprocessing import remove_stopwords

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

    trainingArguments= train_df["Argument1"].append(train_df["Argument2"])

    # Create sentiment analysis extractor
    sentiment= SentimentExtractor()

    # Predict sentiments for all dataset arguments
    X1_train_sentiment= train_df["Argument1"].apply(sentiment.predictSentiment).tolist()
    X2_train_sentiment= train_df["Argument2"].apply(sentiment.predictSentiment).tolist()
    X1_eval_sentiment= eval_df["Argument1"].apply(sentiment.predictSentiment).tolist()
    X2_eval_sentiment= eval_df["Argument2"].apply(sentiment.predictSentiment).tolist()

    # Transform predictions into one-hot encodings
    X1_train_sentiment= pd.get_dummies(X1_train_sentiment)
    X2_train_sentiment= pd.get_dummies(X2_train_sentiment)
    X1_eval_sentiment= pd.get_dummies(X1_eval_sentiment)
    X2_eval_sentiment= pd.get_dummies(X2_eval_sentiment)

    # Combine sentiment predictions of tweets in each pair
    X_train = pd.concat([X1_train_sentiment,X2_train_sentiment] , axis=1).to_numpy()
    X_eval = pd.concat([X1_eval_sentiment,X2_eval_sentiment], axis =1).to_numpy()

    Y_train= train_df["Label"]
    Y_eval= eval_df["Label"]

    # Create LR model and fit to training data
    model = LogisticRegression(solver='liblinear', random_state=0).fit(X_train, Y_train)

    # Evaluate model on evaluation data
    eval_predicted= model.predict(X_eval)
    f1= f1_score(Y_eval, eval_predicted)
    acc= accuracy_score(Y_eval, eval_predicted)
    eval_probs_predicted= model.predict_proba(X_eval)[:,1]
    Y_eval= Y_eval.to_numpy(dtype=np.float64)
    loss= log_loss(Y_eval, eval_probs_predicted)

    # Print evaluation results
    print('F1=', f1, ', Accuracy=', acc, ', Loss=', loss)
    # F1= 0.6238532110091742 , Accuracy= 0.4605263157894737 , Loss= 0.7560372589459589