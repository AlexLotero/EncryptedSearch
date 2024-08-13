#!/usr/bin/env python3

import csv
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def build_model():
    bbc_data = pd.read_csv("data/bbc_data.csv")
    X = bbc_data['data']
    y = bbc_data['labels']
    # print(bbc_data.head())
    # print(bbc_data['data'][1])
    # exit()
    # scale the training data
    
    # train-test split - write here
    X_train, X_test, y_no_train, y_no_test = train_test_split(X, y, test_size=0.4)
    print(len(X_train), len(X_test), len(y_no_train))
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