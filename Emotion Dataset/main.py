"""
Author: gorgeousdays@outlook.com
Date: 2022-03-22 16:37:30
LastEditTime: 2022-03-22 19:10:17
Summary: Sentiment Analysis  DataSet:https://github.com/dair-ai/emotion_dataset
"""


import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import os
import time
import warnings
import numpy as np
import pandas as pd


np.random.seed(2022)
warnings.filterwarnings("ignore")

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence
from keras.models import Model, Sequential,load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, BatchNormalization


# Config
emotions = [ "sadness", "joy", "love", "anger", "fear", "surprise"]

label2int = {
  "sadness": 0,
  "joy": 1,
  "love": 2,
  "anger": 3,
  "fear": 4,
  "surprise": 5
}

max_features = 10000 
max_len = 200 
batch_size=128
epochs=6

def create_LSTMmodel(max_len,max_features,embedding_neurons=128,lstm_neurons=64):

    sequence = Input(shape=(max_len,), dtype='int32')
    embedded = Embedding(max_features, embedding_neurons, input_length=max_len)(sequence)
    bnorm = BatchNormalization()(embedded)
    forwards = LSTM(lstm_neurons, dropout_W=0.2, dropout_U=0.2)(bnorm)
    after_dp = Dropout(0.5)(forwards)
    output = Dense(6, activation='softmax')(after_dp)

    return Model(input=sequence, output=output)


def load_from_pickle(directory):
    return pickle.load(open(directory,"rb"))

def load_data(saveData=False,datapath="merged_training.pkl",samplesize=20000):
    data = load_from_pickle(datapath)
    
    
    data= data[data["emotions"].isin(emotions)]
    data = data.sample(n=samplesize)
    print("data size:",len(data))
    
    data.reset_index(drop=True, inplace=True)

    
    X_train,X_val,y_train,y_val=train_test_split(data.text.to_numpy(), data.emotions.to_numpy(), test_size=0.2)
    X_val,X_test,y_val,y_test=train_test_split(X_val, y_val, test_size=0.5)
    print("Train data:",len(X_train),",Val Data:",len(X_val),",Test Data:",len(X_test))
    
    if saveData:
        train_dataset = pd.DataFrame(data={"text": X_train, "class": y_train})
        val_dataset = pd.DataFrame(data={"text": X_val, "class": y_val})
        test_dataset = pd.DataFrame(data={"text": X_test, "class": y_test})
        final_dataset = {"train": train_dataset, "val": val_dataset , "test": test_dataset }

        train_path = "train.txt"
        test_path = "test.txt"
        val_path = "val.txt"
        
        train_dataset.to_csv(train_path, sep=";",header=False, index=False)
        val_dataset.to_csv(test_path, sep=";",header=False, index=False)
        test_dataset.to_csv(val_path, sep=";",header=False, index=False)
        
        
    y_train=np.array([label2int[i] for i in y_train])
    y_val=np.array([label2int[i] for i in y_val])
    y_test=np.array([label2int[i] for i in y_test])

    # one-hot encode
    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(y_train)
    y_val = encoder.fit_transform(y_val)
    y_test = encoder.fit_transform(y_test)

    return X_train,X_val,X_test,y_train,y_val,y_test


def train(evaluate=True,savemodel=False):
    X_train,X_val,X_test,y_train,y_val,y_test=load_data()

    emotionTokenizer = Tokenizer(nb_words=max_features)
    emotionTokenizer.fit_on_texts(X_train)
    
    #TODO stop words should be used
    intToWord = {}
    for word, value in emotionTokenizer.word_index.items():
        intToWord[value] = word

    intToWord[0] = "NA"

    X_train = emotionTokenizer.texts_to_sequences(X_train)
    X_val=emotionTokenizer.texts_to_sequences(X_val)
    X_test = emotionTokenizer.texts_to_sequences(X_test)
    
    X_train = sequence.pad_sequences(X_train, maxlen=max_len)
    X_val=sequence.pad_sequences(X_val,maxlen=max_len)
    X_test = sequence.pad_sequences(X_test, maxlen=max_len)

    model=create_LSTMmodel(max_len,max_features)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    print('Train...')
    start_time = time.time()
    
    model.fit(X_train, y_train,
        batch_size=batch_size,
        nb_epoch=epochs,
        validation_data=[X_val, y_val], 
        verbose=2)
        
    end_time = time.time()
    average_time_per_epoch = (end_time - start_time) / epochs
    print("avg sec per epoch:", average_time_per_epoch)

    if savemodel:
        model.save("model.pt")

    if evaluate:
        y_pred=model.predict(X_test)
        # Change the y_pred to one-hot
        for i in range(len(y_pred)):
            max_value=max(y_pred[i])
            for j in range(len(y_pred[i])):
                if max_value==y_pred[i][j]:
                    y_pred[i][j]=1
                else:
                    y_pred[i][j]=0
        print(classification_report(y_test, y_pred, target_names=label2int.keys(), digits=len(emotions)))    


def test():
    X_train,X_val,X_test,y_train,y_val,y_test=load_data()
    imdbTokenizer = Tokenizer(nb_words=max_features)
    imdbTokenizer.fit_on_texts(X_train)

    testtext="Classic 40 year old Soho landmark with a burger on pita that is simply the best."
    testtext=imdbTokenizer.texts_to_sequences([testtext])
    testtext = sequence.pad_sequences(testtext, maxlen=max_len)
    print(model.predict(testtext))


def main():
    train(savemodel=True)
    #test()


if __name__ == '__main__':
    main()
