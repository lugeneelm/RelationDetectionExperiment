from Preprocessing import preprocess
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, log_loss
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    # Extract dataset from csv file
    df= pd.read_csv("../Datasets/PAIRED-ANNOTATED.csv")

    # Preprocess tweet text
    df["Argument1"]= df["Argument1"].apply(preprocess)
    df["Argument2"]= df["Argument2"].apply(preprocess)

    # Split dataset into training and evaluation, 80-20
    trainingSize=int(np.rint(df.shape[0]*0.8))
    df.dropna(inplace=True)
    train_df= df[:trainingSize]
    eval_df= df[trainingSize:]

    # Fit count vectoriser on training arguments
    trainingArguments= train_df["Argument1"].append(train_df["Argument2"])
    vectorizer = CountVectorizer(min_df=4)
    vectorizer.fit(trainingArguments)

    # Vectorise arguments
    A1_train= vectorizer.transform(train_df["Argument1"]).toarray()
    A2_train= vectorizer.transform(train_df["Argument2"]).toarray()

    A1_eval= vectorizer.transform(eval_df["Argument1"]).toarray()
    A2_eval= vectorizer.transform(eval_df["Argument2"]).toarray()

    # Combine vectorised tweets in each pair
    X_train= np.column_stack((A1_train, A2_train))
    X_eval= np.column_stack((A1_eval,A2_eval))

    Y_train= train_df["Label"]
    Y_eval= eval_df["Label"]

    # Create Logistic Regression model and fit on training data
    model = LogisticRegression(solver='liblinear', C=0.45).fit(X_train, Y_train)

    # Evaluate LR model on evaluation data
    eval_predicted= model.predict(X_eval)
    f1= f1_score(Y_eval, eval_predicted)
    acc= accuracy_score(Y_eval, eval_predicted)
    eval_probs_predicted= model.predict_proba(X_eval)[:,1]
    loss= log_loss(Y_eval, eval_probs_predicted)

    # Print results of evaluation
    print('F1=', f1, ', Accuracy=', acc, ', Loss=', loss)
    # F1= 0.7311827956989247 , Accuracy= 0.6710526315789473 , Loss= 0.8436633211608846