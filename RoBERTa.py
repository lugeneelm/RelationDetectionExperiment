from simpletransformers.classification import ClassificationModel
import pandas as pd
import sklearn
from Preprocessing import preprocess
import numpy as np

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
    eval_df= df[trainingSize:].reset_index(drop=True)
    
    Y_train= train_df["labels"].to_list()
    Y_eval= eval_df["labels"].to_list()

    # Specift parameters for training
    training_arguments={
        "max_seq_length": 250,
        "train_batch_size": 32,
        "num_train_epochs": 10,
        "overwrite_output_dir": True
    }

    # Create a RoBERTa classification model
    model = ClassificationModel('roberta', 'roberta-base', num_labels=2, use_cuda=False, args= training_arguments)

    # Finetune the model using training data
    model.train_model(train_df)

    # Evaluate the model and print results
    result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score, f1=sklearn.metrics.f1_score)
    print(result)

    # {'mcc': 0.15733517752775894, 'tp': 29, 'tn': 14, 'fp': 25, 'fn': 8, 'auroc': 0.6223146223146223, 'auprc': 0.6055887579246484, 'acc': 0.5657894736842105, 'f1': 0.6373626373626374, 'eval_loss': 0.7141973793506622}