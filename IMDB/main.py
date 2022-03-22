"""
Author: gorgeousdays@outlook.com
Date: 2022-03-19 10:00:12
LastEditTime: 2022-03-22 19:09:24
Summary: Sentiment Analysis
"""
import os
import time
import warnings
import numpy as np

np.random.seed(2022)
warnings.filterwarnings("ignore")

from keras.preprocessing import sequence
from keras.models import Model, Sequential,load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, BatchNormalization
from keras.datasets import imdb

import os
from keras.preprocessing.text import Tokenizer


max_features = 10000
max_len = 200 
batch_size=32
epochs=6


def create_LSTMmodel(max_len,max_features,embedding_neurons=128,lstm_neurons=64):

    sequence = Input(shape=(max_len,), dtype='int32')
    embedded = Embedding(max_features, embedding_neurons, input_length=max_len)(sequence)
    bnorm = BatchNormalization()(embedded)
    forwards = LSTM(lstm_neurons, dropout_W=0.2, dropout_U=0.2)(bnorm)
    after_dp = Dropout(0.5)(forwards)
    output = Dense(1, activation='sigmoid')(after_dp)

    return Model(input=sequence, output=output)

def load_data(datapath):
    train_path=datapath+'/train'
    test_path=datapath+'/test'

    X_train,y_train,X_test,y_test=[],[],[],[]

    X_train.extend([open(train_path+'/pos/' + f,encoding='utf-8').read() for f in os.listdir(train_path+'/pos' ) if f.endswith('.txt')])
    y_train.extend([1 for _ in range(12500)])

    X_train.extend([open(train_path+'/neg/' + f,encoding='utf-8').read() for f in os.listdir(train_path+'/neg') if f.endswith('.txt')])
    y_train.extend([0 for _ in range(12500)])

    X_test.extend([open(test_path +'/pos/'+ f,encoding='utf-8').read() for f in os.listdir(test_path +'/pos') if f.endswith('.txt')])
    y_test.extend([1 for _ in range(12500)])

    X_test.extend([open(test_path +'/neg/'+ f,encoding='utf-8').read() for f in os.listdir(test_path +'/neg') if f.endswith('.txt')])
    y_test.extend([0 for _ in range(12500)])

    return X_train,y_train,X_test,y_test

def train():
    X_train,y_train,X_test,y_test=load_data('./aclImdb')

    imdbTokenizer = Tokenizer(nb_words=max_features)

    imdbTokenizer.fit_on_texts(X_train)
    
    #TODO: stop words should be used
    intToWord = {}
    for word, value in imdbTokenizer.word_index.items():
        intToWord[value] = word

    intToWord[0] = "NA"

    X_train = imdbTokenizer.texts_to_sequences(X_train)
    X_test = imdbTokenizer.texts_to_sequences(X_test)

    X_train = sequence.pad_sequences(X_train, maxlen=max_len)
    X_test = sequence.pad_sequences(X_test, maxlen=max_len)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model=create_LSTMmodel(max_len,max_features)

    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    print('Train...')
    start_time = time.time()

    model.fit(X_train, y_train,
                batch_size=batch_size,
                nb_epoch=epochs,
                validation_data=[X_test, y_test], 
                verbose=2)

    end_time = time.time()
    average_time_per_epoch = (end_time - start_time) / epochs
    print("avg sec per epoch:", average_time_per_epoch)

    model.save("model.pt")

def test():

    X_train,y_trian,X_test,y_test=load_data('./aclImdb')
    imdbTokenizer = Tokenizer(nb_words=max_features)
    imdbTokenizer.fit_on_texts(X_train)

    model=load_model("model.pt")
    
    testtext="Classic 40 year old Soho landmark with a burger on pita that is simply the best."

    testtext=imdbTokenizer.texts_to_sequences([testtext])

    testtext = sequence.pad_sequences(testtext, maxlen=max_len)

    print(model.predict(testtext))


def main():
    train()
    #test()

if __name__ == '__main__':
    main()