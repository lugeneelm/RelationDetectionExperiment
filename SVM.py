from sklearn import svm
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, log_loss
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from Preprocessing import preprocess

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

    # Create Support Vector Machine model and fit on training data
    model=svm.SVC(kernel='linear', C=0.1)
    model.fit(X_train, Y_train)

    # Create calibrated classifier to get probabilistic predictions
    clf= CalibratedClassifierCV(model).fit(X_train, Y_train)

    # Evaluate SVM model using evaluation dataset
    eval_probs_predicted = clf.predict_proba(X_eval)[:,1]
    eval_predicted=model.predict(X_eval)
    f1= f1_score(Y_eval, eval_predicted)
    acc= accuracy_score(Y_eval, eval_predicted)
    Y_eval= Y_eval.to_numpy(dtype=np.float64)
    loss= log_loss(Y_eval, eval_probs_predicted)

    # Print evaluation results
    print('F1=', f1, ', Accuracy=', acc, ', Loss=', loss)
    # F1= 0.7234042553191489 , Accuracy= 0.6578947368421053 , Loss= 0.7140861130332254