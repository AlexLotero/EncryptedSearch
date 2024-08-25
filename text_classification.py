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

# Function to perform preproccessing on the data, including: 
# - tokenization to convert raw text into a sequence of integers,
# - padding to make all inputs the same size, and
# - encoding to represent each unique label as a unique integer
def preprocessor(X_tr_p, y_tr_p, X_cv_p, y_cv_p, X_te_p, y_te_p, max_len, num_word):
    
    # Tokenizing the data: each unique word --> a unique integer
    tokenizer = Tokenizer(num_words = num_word)
    tokenizer.fit_on_texts(X_tr_p)
    X_train_sq = tokenizer.texts_to_sequences(X_tr_p)
    X_cv_sq = tokenizer.texts_to_sequences(X_cv_p)
    X_test_sq = tokenizer.texts_to_sequences(X_te_p)

    # Pad the data to ensure a consistent size for inputs
    X_train = pad_sequences(X_train_sq, maxlen=max_len)
    X_cv = pad_sequences(X_cv_sq, maxlen=max_len)
    X_test = pad_sequences(X_test_sq, maxlen=max_len)


    # Combine all labels for splitting
    all_labels = np.concatenate([y_tr_p, y_cv_p, y_te_p])

    # "Fit" LabelEncoder on all labels - define the integers needed for encoding
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    # Transform (encode) the labels from strings to unique integers
    y_train = label_encoder.transform(y_tr_p)
    y_cv = label_encoder.transform(y_cv_p)
    y_test = label_encoder.transform(y_te_p)
    
    return X_train, y_train, X_cv, y_cv, X_test, y_test

# Funtion to create, compare, and save our model to disk
def build_model():
    # List containing possible num_word values for tokenization to test
    num_words = [10000, 15000, 20000]

    # Define several empty lists for storage models and their related loss values, X (input) values, and y (output) values
    abs_models = []
    abs_losses = []
    abs_X = []
    abs_y = []

    # Outer loop to try the different num_word values for tokenization defined above
    for word_num in num_words:
        # Read in training data courtesy of Kaggle: https://www.kaggle.com/datasets/alfathterry/bbc-full-text-document-classification/data
        # Assign article titles and their categories to input (X) and output (y) lists, respectively 
        bbc_data = pd.read_csv("data/bbc_data.csv")
        X = bbc_data['data']
        y = bbc_data['labels']

        # Define important variables for use later in our preproccessing phase
        num_classes = len(np.unique(y))
        max_words = 0
        X_lengths = []
        for x in X:
            if len(x.split()) > max_words:
                max_words = len(x.split())
            X_lengths.append(len(x.split()))
        n_percentile = int(np.percentile(X_lengths, 90))
        
        # Split data into three sets: training to initially train a model, 
        # cross validation to compare models within the inner loop, and 
        # testing to compare models within the oputer loop
        # Test set:
        X_train_p, X_no_test, y_train_p, y_no_test = train_test_split(X, y, 
                                                                      test_size=0.4, 
                                                                      shuffle=True, 
                                                                      stratify=y)

        # Cross validation set:
        X_cv_p, X_test_p, y_cv_p, y_test_p = train_test_split(X_no_test, 
                                                              y_no_test, 
                                                              test_size = 0.5, 
                                                              shuffle=True, 
                                                              stratify=y_no_test)
        
        # Test set:
        X_train, y_train, X_cv, y_cv, X_test, y_test = preprocessor(X_train_p, y_train_p, 
                                                                    X_cv_p, y_cv_p,
                                                                    X_test_p, y_test_p, 
                                                                    n_percentile, word_num)

        # Create Lists to store models and the ouput of their loss functions, respectively
        models = []
        losses = []
        # accuracys = []

        # Create a variable to store the size of the input to the first layer of the neural network
        input_shape = X_train.shape[1]

        # Define a size for the final hidden layer
        hid_layer = 16

        # Inner loop to test different model complexities with number/size of hidden layers
        for i in range(1, 9):
            model = Sequential()

            # Define the input layer
            model.add(Dense(units=(hid_layer+(hid_layer // 2)*i), 
                                    activation='relu', 
                                    input_shape=(input_shape, )))
            
            # Populate the network with hidden layers
            for j in range(i-1, 0, -1):
                model.add(Dense(units=(hid_layer+(hid_layer // 2)*j), activation='relu'))

            # Assign the output layer with as many units as there are categories in the training data
            # Utilizes the 'linear' activation function in order to deploy the imporved softmax implementation
            model.add(Dense(units=num_classes, activation='linear'))

            # Compile the model with the Adam optimizer, the SCCE loss function, and the improved softmax implementation
            model.compile(optimizer=Adam(learning_rate=0.01),
                          #metrics=['accuracy'],
                          loss=SparseCategoricalCrossentropy(from_logits=True))
            
            # Fit the model to our training set
            model.fit(X_train, y_train, epochs=100, verbose=0)
            #model.summary()

            # Store the current model and its loss value in a list for reference later
            # loss, accuracy = model.evaluate(X_cv, y_cv)
            loss = model.evaluate(X_cv, y_cv)
            models.append(model)
            losses.append(loss)
            # accuracys.append(accuracy)

        # Find the lowest recorded loss by an individual model
        min_value = min(losses)

        # Get the index of the minimum value
        min_index = losses.index(min_value)

        # print("Model: " + models[min_index].summary())
        # print("Loss: " + losses[min_index])
        # print("Accuracy: " + accuracys[min_index])

        # Store the important values of the best performing model from this num_words iteration
        abs_models.append(models[min_index])
        abs_losses.append(losses[min_index])
        abs_X.append(X_test)
        abs_y.append(y_test)

    # Identify which value for num_words produced the best performing model and store it in a variable
    nn_loss = 10
    for nn_model in range(len(abs_models)):
            this_loss = abs_models[nn_model].evaluate(abs_X[nn_model], abs_y[nn_model])
            if this_loss < nn_loss:
                final_model = abs_models[nn_model]
                final_loss = this_loss
                nn_loss = this_loss
    
    # Print the details of the best performing model
    print()
    print("The final model was selected as follows: \n")
    print(final_model.summary())
    print("Final Loss: %f" % final_loss)
    print()

    # Save the model to disc for use within fencrypt
    model_stored = "models/text_classification_model.keras"
    model.save(model_stored)
    return model_stored

def main():
    # Begin the timer to report runtime
    start_time = time.time()

    # Save the model's location by starting the model selection process with a call to the build model function
    model_location = build_model()

    # Report the time taken to produce the model and the location at which it has been saved to disc
    print("Text Classification Model built and available at %d" + model_location)
    with open("model_location.txt", "w") as file:
        file.write(model_location)
    print("Process Runtime: %s minutes " % ((time.time() - start_time)/60))

if __name__ == "__main__":
    main()