from sklearn.metrics import f1_score, log_loss, accuracy_score
import pandas as pd
from tensorflow.keras.layers import Dense, Embedding, LSTM, Concatenate, TextVectorization, Dropout
from tensorflow.keras.models import Model
from keras import backend as K
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization

# The following 3 methods have been sourced from https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
# Create metric used for evaluation since unavailable with keras model
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

if __name__ == '__main__':
    # Extract dataset from csv file
    df= pd.read_csv("../Datasets/PAIRED-ANNOTATED.csv")

    # Split dataset into training and evaluation, 80-20
    trainingSize=int(np.rint(df.shape[0]*0.8))
    df.dropna(inplace=True)
    train_df= df[:trainingSize]
    eval_df= df[trainingSize:].reset_index(drop=True)

    # Tokenize tweet pairs using Bag Of Words method
    trainingArguments= train_df["Argument1"].append(train_df["Argument2"]).to_numpy()

    # code to create Embedding layer from https://github.com/keras-team/keras-io/blob/cc6cb93792c062b077a091f48de8d29db75791a0/examples/nlp/pretrained_word_embeddings.py
    # Create text vectoriser and fit to training arguments
    vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=200)
    vectorizer.adapt(trainingArguments)

    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))

    path_to_glove_file = "../Datasets/glove-2/glove.6B.100d.txt"

    embeddings_index = {}
    with open(path_to_glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    num_tokens = len(voc) + 2
    embedding_dim = 100
    hits = 0
    misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1

    embedding_layer1 = Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,
    )

    embedding_layer2 = Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,
    )

    # Create Multi-layer model
    A1 = keras.Input(shape=(None,), dtype="float64")
    A2 = embedding_layer1(A1)
    A3= LSTM(32)(A2)
    A4= Dropout(0.2)(A3)

    B1 = keras.Input(shape=(None,), dtype="float64")
    B2 = embedding_layer2(B1)
    B3= LSTM(32)(B2)
    B4= Dropout(0.2)(B3)

    concatted = Concatenate(axis=1)([A4, B4])
    M1 = Dense(16, activation='relu')(concatted)
    M2= Dense(1, activation='sigmoid')(M1)

    # Define model input and output layers
    model = Model(inputs=[A1,B1], outputs=M2)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])

    Y_train= train_df["Label"]
    Y_eval= eval_df["Label"]

    # Vectorise training and evaluation datasets
    X1_train = vectorizer(np.array([[s] for s in train_df['Argument1']])).numpy()
    X2_train = vectorizer(np.array([[s] for s in train_df['Argument2']])).numpy()

    X1_eval = vectorizer(np.array([[s] for s in eval_df['Argument1']])).numpy()
    X2_eval = vectorizer(np.array([[s] for s in eval_df['Argument2']])).numpy()

    # Uncomment to display diagram of multilayer model produced
    # plot_model(model,to_file="model.png", show_layer_names=False)

    # Train model
    model.fit([X1_train, X2_train], Y_train, epochs=200, validation_split=0.1)

    # Evaluate model 
    y_predict = model.predict([X1_eval, X2_eval], batch_size=1)
    loss= log_loss(Y_eval, y_predict)
    y_predict = np.round(y_predict)
    y_predict = y_predict.astype(np.float64)
    f1= f1_score(Y_eval, y_predict)
    acc= accuracy_score(Y_eval, y_predict)

    # Print evaluation results
    print('F1=', f1, ', Accuracy=', acc, ', Loss=', loss)
    #  F1= 0.6548672566371682 , Accuracy= 0.4868421052631579 , Loss= 0.7142376115447596
