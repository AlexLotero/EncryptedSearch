#!/usr/bin/env python3

import time
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

def preprocessor(X_tr_p, y_tr_p, X_cv_p, y_cv_p, X_te_p, y_te_p, max_len, num_word):
    tokenizer = Tokenizer(num_words = num_word)
    tokenizer.fit_on_texts(X_tr_p)

    X_train_sq = tokenizer.texts_to_sequences(X_tr_p)
    X_cv_sq = tokenizer.texts_to_sequences(X_cv_p)
    X_test_sq = tokenizer.texts_to_sequences(X_te_p)

    X_train = pad_sequences(X_train_sq, maxlen=max_len)
    X_cv = pad_sequences(X_cv_sq, maxlen=max_len)
    X_test = pad_sequences(X_test_sq, maxlen=max_len)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_tr_p)
    y_cv = label_encoder.fit_transform(y_cv_p)
    y_test = label_encoder.fit_transform(y_te_p)
    
    return X_train, y_train, X_cv, y_cv, X_test, y_test

def build_model():
    num_words = [10000, 15000, 20000]
    for word_num in num_words:
        bbc_data = pd.read_csv("data/bbc_data.csv")
        X = bbc_data['data']
        y = bbc_data['labels']
        max_words = 0
        X_lengths = []
        for x in X:
            if len(x.split()) > max_words:
                max_words = len(x.split())
            X_lengths.append(len(x.split()))
        n_percentile = int(np.percentile(X_lengths, 90))
        # print(n_percentile)
        # exit()
        # print(len(np.unique(y)))
        # exit()
        # print(bbc_data.head())
        # print(bbc_data['data'][1])
        # exit()
        # scale the training data
        
        # train-test split - write here
        X_train_p, X_no_test, y_train_p, y_no_test = train_test_split(X, y, test_size=0.4)
        X_cv_p, X_test_p, y_cv_p, y_test_p = train_test_split(X_no_test, y_no_test, test_size = 0.5)
        X_train, y_train, X_cv, y_cv, X_test, y_test = preprocessor(X_train_p, y_train_p, 
                                                                    X_cv_p, X_test_p, 
                                                                    y_cv_p, y_test_p, 
                                                                    n_percentile, num_words)
        models = []
        hid_layer = 64
        for i in range(1, 9):
            print("i = ", i)
            model = Sequential()
            for j in range(i, 1, -1):
                print('j = ', (hid_layer+(hid_layer/2)*j))
                model.add(Dense(units=(hid_layer+(hid_layer/2)*j), activation='relu'))
            model.add(Dense(units=len(np.unique(y)), activation='linear'))
            model.compile(optimizer=Adam(learning_rate=0.01), 
                    loss=SparseCategoricalCrossentropy(from_logits=True))
            model.fit(X_train, y_train, epochs=100)
            # loss, accuracy = model.evaluate(X_cv, y_cv)
            # print(loss, accuracy)

        model_stored = "/models"
        return model_stored

def main():
    start_time = time.time()
    model_location = build_model()
    # model_location = "EncryptedSearch/models/"
    print("Text Classification Model built and available at %d", model_location)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()