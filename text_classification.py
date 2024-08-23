#!/usr/bin/env python3

# https://www.kaggle.com/datasets/alfathterry/bbc-full-text-document-classification/data

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
    # Convert Series to list
    # X_tr_p = X_tr_p.tolist()
    # X_cv_p = X_cv_p.tolist()
    # X_te_p = X_te_p.tolist()

    # print("Type and sample of X_tr_p before processing:", type(X_tr_p[0]), X_tr_p[:3])
    # print("Type and sample of X_cv_p before processing:", type(X_cv_p[0]), X_cv_p[:3])
    # print("Type and sample of X_te_p before processing:", type(X_te_p.tolist()[0]), X_te_p.tolist()[:3])
    # exit()
    
    tokenizer = Tokenizer(num_words = num_word)
    tokenizer.fit_on_texts(X_tr_p)

    X_train_sq = tokenizer.texts_to_sequences(X_tr_p)
    X_cv_sq = tokenizer.texts_to_sequences(X_cv_p)
    # for z in X_te_p:
    #     if not isinstance(z, str):
    #         print(z)
    #         exit()
    # print(X_te_p, len(X_te_p))
    # X_te_p = X_te_p.tolist()
    # print("Type and sample of X_te_p before processing:", type(X_te_p[0]), X_te_p[:3])
    # exit()

    X_test_sq = tokenizer.texts_to_sequences(X_te_p)

    X_train = pad_sequences(X_train_sq, maxlen=max_len)
    X_cv = pad_sequences(X_cv_sq, maxlen=max_len)
    X_test = pad_sequences(X_test_sq, maxlen=max_len)


    # Combine all labels before splitting
    all_labels = np.concatenate([y_tr_p, y_cv_p, y_te_p])

    # Fit LabelEncoder on all labels
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    # Transform the labels separately after splitting
    y_train = label_encoder.transform(y_tr_p)
    y_cv = label_encoder.transform(y_cv_p)
    y_test = label_encoder.transform(y_te_p)

    # print("Unique values in y_train:", np.unique(y_train), len(np.unique(y_train)), type(y_train))
    # print("Unique values in y_cv:", np.unique(y_cv), len(np.unique(y_cv)), type(y_cv))
    # print("Unique values in y_test:", np.unique(y_test), len(np.unique(y_test)), type(y_test))
    # exit()
    
    return X_train, y_train, X_cv, y_cv, X_test, y_test

def build_model():
    num_words = [10000, 15000, 20000]
    for word_num in num_words:
        bbc_data = pd.read_csv("data/bbc_data.csv")

        # cnt = 0
        # for item in range(len(bbc_data["data"])):
        #     if not isinstance(bbc_data['data'][item], str):
        #         print(bbc_data['data'][item], bbc_data['labels'][item])
        #         cnt += 1
        #     elif item % 20:
        #         print(bbc_data['data'][item], bbc_data['labels'][item])
        # print(cnt)
        # exit()

        X = bbc_data['data']
        y = bbc_data['labels']
        num_classes = len(np.unique(y))
        # label_encoder = LabelEncoder()
        # y_enc = label_encoder.fit_transform(y)
        # print(np.unique(y), len(np.unique(y)))
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
        X_train_p, X_no_test, y_train_p, y_no_test = train_test_split(X, y, test_size=0.4, shuffle=True, stratify=y)
        # X_train_p, X_no_test, y_train_p, y_no_test = train_test_split(X, y_enc, test_size=0.4, shuffle=True, stratify=y)
        
        # X_train_p = X_train_p.tolist()
        # X_no_test = X_no_test.tolist()

        # print("Type of X_train_p after splitting:", type(X_train_p.tolist()[0]))
        # print("Type of X_no_test after splitting:", type(X_no_test.tolist()[0]))

        X_cv_p, X_test_p, y_cv_p, y_test_p = train_test_split(X_no_test, y_no_test, test_size = 0.5, shuffle=True, stratify=y_no_test)
        
        # X_test_p = X_test_p.tolist()
        # print("Type of X_test_p after second split:", type(X_test_p.tolist()[0]))
        # exit()

        # print(y_train_p)
        # print(y_cv_p)
        # print(y_test_p)
        # print("Unique values in y_train:", np.unique(y_train_p), len(np.unique(y_train_p)))
        # print("Unique values in y_cv:", np.unique(y_cv_p), len(np.unique(y_cv_p)))
        # print("Unique values in y_test:", np.unique(y_test_p), len(np.unique(y_test_p)))
        # exit()
        X_train, y_train, X_cv, y_cv, X_test, y_test = preprocessor(X_train_p, y_train_p, 
                                                                    X_cv_p, y_cv_p,
                                                                    X_test_p, y_test_p, 
                                                                    n_percentile, word_num)
        
        # print("Unique values in y_train:", np.unique(y_train), len(np.unique(y_train)), type(y_train))
        # print("Unique values in y_cv:", np.unique(y_cv), len(np.unique(y_cv)), type(y_cv))
        # print("Unique values in y_test:", np.unique(y_test), len(np.unique(y_test)), type(y_test))
        # exit()

        models = []
        losses = []
        accuracys = []
        input_shape = X_train.shape[1]
        
        # print(input_shape)
        # print("X_train shape:", X_train.shape)
        # print("X_cv shape:", X_cv.shape)
        # print("Type of X_train:", type(X_train))
        # print("Type of X_cv:", type(X_cv))
        # exit()

        hid_layer = 16
        for i in range(1, 9):
            print("i = ", i)
            model = Sequential()
            # input_shape=(input_shape,)
            model.add(Dense(units=(hid_layer+(hid_layer // 2)*2), 
                                    activation='relu', 
                                    input_shape=(input_shape, )))
            for j in range(i-1, 0, -1):
                # if j == i:
                #     model.add(Dense(units=(hid_layer+(hid_layer // 2)*j), 
                #                     activation='relu', 
                #                     input_shape=(input_shape,)))
                # # print('j = ', (hid_layer+(hid_layer/2)*j))
                # else:
                model.add(Dense(units=(hid_layer+(hid_layer // 2)*j), activation='relu'))
            model.add(Dense(units=num_classes, activation='softmax'))
            model.compile(optimizer=Adam(learning_rate=0.01), 
                    loss=SparseCategoricalCrossentropy(from_logits=True))
            
            model.fit(X_train, y_train, epochs=100)
            model.summary()
            # print("Model output units:", model.output_shape[-1])
            # print("Number of classes:", len(np.unique(y_train)))
            # print("y_train unique: ", np.unique(y_train))
            # exit()

            # loss, accuracy = model.evaluate(X_cv, y_cv)
            loss = model.evaluate(X_cv, y_cv)
            models.append(model)
            losses.append(loss)
            # accuracys.append(accuracy)
            # print(loss, accuracy)
        print()
        print("Models: " + models)
        print("Losses: " + losses)
        # print("Accuracys: " + accuracys)
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