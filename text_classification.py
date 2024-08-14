#!/usr/bin/env python3

import csv
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

def build_model():
    bbc_data = pd.read_csv("data/bbc_data.csv")
    X = bbc_data['data']
    y = bbc_data['labels']
    # print(len(np.unique(y)))
    # exit()
    # print(bbc_data.head())
    # print(bbc_data['data'][1])
    # exit()
    # scale the training data
    
    # train-test split - write here
    X_train, X_no_test, y_train, y_no_test = train_test_split(X, y, test_size=0.4)
    X_cv, X_test, y_cv, y_test = train_test_split(X_no_test, y_no_test, test_size = 0.5)
    models = []
    hid_layer = 64
    for i in range(1, 9):
        # print("i = ", i)
        model = Sequential()
        for j in range(i, 1, -1):
            print('j = ', j)
            model.add(Dense(units=(hid_layer+(hid_layer/2)*j), activation='relu'))
        model.add(Dense(units=len(np.unique(y)), activation='linear'))
    exit()

    # print(len(X_train), len(X_test), len(y_no_train))
    exit()
    

    model.compile(loss=SparseCategoricalCrossEntropy(from_logits=True))
    model_stored = "/models"
    return model_stored

def main():
    model_location = build_model()
    # model_location = "EncryptedSearch/models/"
    print("Text Classification Model built and available at %d", model_location)

if __name__ == "__main__":
    main()