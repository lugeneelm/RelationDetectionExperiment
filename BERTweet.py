from simpletransformers.classification import ClassificationModel
import pandas as pd
import sklearn
import numpy as np
from Preprocessing import preprocess

if __name__ == '__main__':
    # Extract dataset from csv file
    df= pd.read_csv("../Datasets/PAIRED-ANNOTATED.csv")

    # Preprocess tweet text
    df["Argument1"]= df["Argument1"].apply(preprocess)
    df["Argument2"]= df["Argument2"].apply(preprocess)

    # Rename dataset column names for transformer model
    df.columns = ['text_a', 'text_b', 'labels']

    # Split dataset into training and evaluation, 80-20
    trainingSize=int(np.rint(df.shape[0]*0.8))
    df.dropna(inplace=True)
    train_df= df[:trainingSize]
    eval_df= df[trainingSize:]
    
    Y_train= train_df["labels"].to_list()
    Y_eval= eval_df["labels"].to_list()

    # Specify training parameters   
    training_arguments={
        "train_batch_size": 32,
        "num_train_epochs": 20,
        "overwrite_output_dir": True,
    }

    # Create a BERTweet model
    model = ClassificationModel('bertweet', "vinai/bertweet-base", num_labels=2, use_cuda=False, args= training_arguments)

    # Finetune the model on training data
    model.train_model(train_df)

    # Evaluate the model on evaluation dataset and print results
    result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score, f1=sklearn.metrics.f1_score)
    print(result)
    # {'mcc': 0.20297378678399594, 'tp': 28, 'tn': 17, 'fp': 22, 'fn': 9, 'auroc': 0.5855855855855856, 'auprc': 0.5152972767654933, 'acc': 0.5921052631578947, 'f1': 0.6436781609195402, 'eval_loss': 0.9243814289569855}