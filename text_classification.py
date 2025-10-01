#!/usr/bin/env python3

import time
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import json
import pickle

from tensorflow.keras.preprocessing.text import tokenizer_from_json

###### DELETE THIS #########
# tf.random.set_seed(42)
# np.random.seed(42)
############################

MODEL_STORED = "models/text_classification_model.keras"
MODEL_LOCATION = "model_location.txt"
TOKENIZER_STORED = "models/tokenizer.json"
LABEL_ENCODER_STORED = "models/label_encoder.pkl"

# def print_label_encoder_summary(label_encoder):
#     if hasattr(label_encoder, 'classes_'):
#         print("LabelEncoder classes:")
#         for i, class_label in enumerate(label_encoder.classes_):
#             print(f"  {class_label} -> {i}")
#     else:
#         print("LabelEncoder has not been fitted yet.")

def preprocessor_1(X, max_len, tokenizer_1):

    # Tokenizing the data: each unique word --> a unique integer
    X_tok = tokenizer_1.texts_to_sequences(X)

    # Pad the data to ensure a consistent size for inputs
    X_processed = pad_sequences(X_tok, maxlen=max_len)
    print("Number of features per sample:", X_processed.shape[1] if X_processed.ndim > 1 else "N/A")
    return X_processed



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

    # Convert the tokenizer to json for storage on disk
    tokenizer = tokenizer.to_json()

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
    # print(y_test)

    y_train = to_categorical(y_train)
    y_cv = to_categorical(y_cv)
    y_test = to_categorical(y_test)
    # print(y_test)
    # print_label_encoder_summary(label_encoder)
    # exit()
    
    return X_train, y_train, X_cv, y_cv, X_test, y_test, tokenizer, label_encoder

# Funtion to create, compare, and save our model to disk
def build_model():

    ###### DELETE THIS #########
    initial_learning_rate = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
    #############################

    # List containing possible num_word values for tokenization to test
    num_words = [10000, 15000, 20000]

    # Define several empty lists for storage models and their related 
    # loss values, X (input) values, y (output) values, and 90th percentiles of input data.
    abs_models = []
    abs_losses = []
    abs_X = []
    abs_y = []
    ninety_p = []
    tokenizers = []
    label_encoders = []

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

        # Create Lists to store models and the ouput of their loss functions, respectively
        models = []
        losses = []
        accuracys = []
        
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
        X_train, y_train, X_cv, y_cv, X_test, y_test, tokenizer, label_encoder = preprocessor(X_train_p, y_train_p, 
                                                                    X_cv_p, y_cv_p,
                                                                    X_test_p, y_test_p, 
                                                                    n_percentile, word_num)

        # Create a variable to store the size of the input to the first layer of the neural network
        input_shape = X_train.shape[1]

        # Define a size for the final hidden layer
        hid_layer = 16 # 8

        # Inner loop to test different model complexities with number/size of hidden layers
        for i in range(2, 10):
            # Define the input layer
            model = Sequential([Input(shape=(input_shape,))])
            
            # Populate the network with hidden layers
            for j in range(i-1, 0, -1):
                model.add(Dense(units=(hid_layer+(hid_layer)*j), activation='relu'))

            # Assign the output layer with as many units as there are categories in the training data
            # Utilizes the 'linear' activation function in order to deploy the imporved softmax implementation
            model.add(Dense(units=num_classes, activation='linear'))

            # Compile the model with the Adam optimizer, the SCCE loss function, and the improved softmax implementation
            model.compile(optimizer=Adam(learning_rate=lr_schedule), # 0.00095),
                          metrics=['accuracy'],
                          loss=CategoricalCrossentropy(from_logits=True))
            
            ########### DELETE THIS ############
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
            model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping, model_checkpoint], verbose=0)


            # Fit the model to our training set
            # model.fit(X_train, y_train, epochs=100, verbose=0)
            #model.summary()

            # Store the current model and its loss value in a list for reference later
            loss, accuracy = model.evaluate(X_cv, y_cv)
            # loss = model.evaluate(X_cv, y_cv)
            models.append(model)
            losses.append(loss)
            # print(accuracy)
            # exit()
            accuracys.append(accuracy)

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
        ninety_p.append(n_percentile)
        tokenizers.append(tokenizer)
        label_encoders.append(label_encoder)

    # Identify which value for num_words produced the best performing model and store it in a variable
    nn_loss = 10
    # final_num_words = 0
    for nn_model in range(len(abs_models)):
            this_loss = abs_models[nn_model].evaluate(abs_X[nn_model], abs_y[nn_model])
            if this_loss < nn_loss:
                final_model = abs_models[nn_model]
                final_loss = this_loss
                nn_loss = this_loss
                # final_num_words = str(num_words[nn_model])
                final_n_percentile = str(ninety_p[nn_model])
                final_tokenizer = tokenizers[nn_model]
                final_label_encoder = label_encoders[nn_model]
    
    # Print the details of the best performing model
    print()
    print("The final model was selected as follows: \n")
    print(final_model.summary())
    foo = final_model.evaluate(X_test, y_test)
    print("HERE:", foo)
    print("Final Loss: %f" % final_loss)
    # print("Most Effective Number of Words %s" % final_num_words)
    print("90th percentile of input text lengths: %s" % final_n_percentile)
    print()

    # Save the model to disc for use within fencrypt
    # model_stored = "models/text_classification_model.keras"
    # model.save(model_stored)
    # return model_stored, final_num_words, final_n_percentile
    
    # final_model.save(MODEL_STORED)
    
    # return final_num_words, final_n_percentile, final_tokenizer

    # for layer in final_model.layers:
    #     weights = layer.get_weights()  # Returns a list of numpy arrays
    #     print(f"Weights for layer {layer.name}:")
    #     for i, weight_array in enumerate(weights):
    #         print(f"Weight {i}:")
    #         print(weight_array)
    # file = open("entertainment_ex.txt", "r")
    # og_file = file.read()
    # file.close()
    og_file = b'Fockers retain film chart crown  Comedy Meet The Fockers has held on to the number one spot at the North American box office for a second week.  It took $42.8m (xc2xa323.7m) at the weekend, making its overall takings more than $163m (xc2xa390m) in 12 days, according to studio estimates. It took $19.1m (xc2xa39.9m) on Christmas Day alone, the highest takings on that day in box office history. The sequel to the Ben Stiller comedy Meet The Parents stars Robert De Niro, Dustin Hoffman and Barbra Streisand. The success of Meet the Fockers could help produce record box office revenue for 2004, said Paul Dergarabedian, president of the industrys tracker Exhibitor Relations. "Weve had a much stronger than anticipated final week of the year that helped the industry end on a high note," said Mr Dergarabedian.  Meet the Fockers also broke the box office records for the most money taken on New Years Eve, when it made $12.2m (xc2xa36.2m), and New Years Day, when it took $18m (xc2xa39.4m). The previous New Years Eve record was set in 2000 by Cast Away with $8.5m (xc2xa34.5m). The Lord of the Rings: The Return of the King had held the New Years Day title with $12.8m (xc2xa36.7m). However, Christmas takings were down 26.5% on 2003s figures - which was blamed on Christmas falling over a weekend this year. This weekends top 12 films took an estimated $125.4m (xc2xa365.8m), a 4.3% increase on the same weekend last year. But there were no major releases last week to provide competition to Meet the Fockers or Lemony Snickets A Series of Unfortunate Events, which finished in second place with $14.7m (xc2xa37.7m). The Aviator, starring Leonardo DiCaprio as Howard Hughes, ended up in third position after taking $11.2m (xc2xa35.9m). Comedy Fat Albert - co-written by Bill Cosby - moved down the chart to fourth place after taking $10.7m (xc2xa35.6m).'
    X_1 = [og_file]
    X_1 = [text.decode('utf-8') for text in X_1]
    og_2 = b'Hodgson shoulders England blame  Fly-half Charlie Hodgson admitted his wayward kicking played a big part in Englands 18-17 defeat to France.  Hodgson failed to convert three penalties and also missed a relatively easy drop goal attempt which would have given England a late win. "Im very disappointed with the result and with my myself," Hodgson said. "It is very hard to take but its something I will have to get through and come back stronger. My trainings been good but it just didnt happen." Hodgson revealed that Olly Barkley had taken three penalties because they were "out of my range" but the centre could not convert his opportunities either, particularly the drop goal late on. "It wasnt a good strike," he added. "I felt as soon as it hit my boot it had missed. Its very disappointing, but I must recover." Andy Robinson said he would "keep working on the kicking" with his squad. However, the England coach added that he would take some positives from the defeat.  "We went out to play and played some very good rugby and what have France done?" he said. "They won the game from kicking penalties from our 10m line. "Its very frustrating. The lads showed a lot of ambition in the first half, they went out to sustain it in the second but couldnt build on it. "We took the ball into contact, and you know when you do that it is a lottery whether the referee is going to give the penalty to your side or the other side. "We have lost a game we should have won. There is a fine line between winning and losing, and for the second week weve been on the wrong side of that line and it hurts."  England went in at half-time with a 17-6 lead but they failed to score in the second half and Dimitri Yachvili slotted over four penalties as France overhauled the deficit. England skipper Jason Robinson admitted his side failed to cope with Frances improved second-half display.  "We controlled the game in the first half but we knew that they would come out and try everything after half-time," he said. "We made a lot of mistakes in the second half and they punished us. They took their chances when they came. "Its very disappointing. Last week we lost by two points, now one point."'
    X_2 = [og_2]
    X_2 = [text.decode('utf-8') for text in X_2]
    og_3 = b"""Adrien Brody attends the "The Brutalist" red carpet during the 81st Venice International Film Festival on September 01, 2024 in Venice, Italy. Brady Corbet's historical drama "The Brutalist," starring Adrien Brody, Felicity Jones and Guy Pearce, wowed Venice Film Festival on Sunday with a 12-minute standing ovation. Brody, who stars in the film as a Hungarian Holocaust survivor struggling to revive his career as an architect in the U.S., was overcome with emotion as the clapping raged on. The actor wiped tears away and held his head in his hands, repeatedly trying to direct the applause toward his director and co-stars, but the spotlight kept falling back on him. The three-and-a-half hour drama, which included a 15-minute intermission, follows Brody's character over nearly four decades as he immigrates to the U.S. and begins working for a rich but hot-headed man who wants to build a community center like no other. He helps Laslav reunite with his ailing wife (Jones) and begins to construct the brutalist building of his dreams, but a fateful incident changes both of their lives forever. The cast also includes Joe Alwyn, Alessandro Nivola, Jonathan Hyde, Isaach De Bankole, Raffey Cassidy, Stacy Martin, Emma Laird and Peter Polycarpou. In addition to directing the film, Corbet also co-wrote the script with his wife Mona Fastvold ("The Sleepwalker"). Actor-turned-director Corbet has premiered films at Venice twice before, with his 2015 directorial debut "The Childhood of a Leader" earning him the Luigi De Laurentiis award for best debut film and the Horizons best director prize and 2018's "Vox Lux" competing for the Golden Lion. "The Brutalist" is also in the running for the festival's prestigious top prize. Teasing "The Brutalist" in an interview with Variety in July, Corbet revealed that the 215-minute, 70mm movie will include an intermission when it screens at Venice. "I like the idea of them," he said. "It gives everyone time to reset and no one has to stress about missing a scene to run to the bathroom, which is a legitimate concern on longer films. I would describe it as a rolling intermission. The movie doesn't stop exactly. There will be images and sound and there is a timer to let the audience know how much time is left."""
    X_3 = [og_3]
    X_3 = [text.decode('utf-8') for text in X_3]
    X_1 = X_1+X_2+X_3
    # # print(type(X), X)
    # # print("FINAL TOKENIZER: ", type(final_tokenizer))
    processed_X = preprocessor_1(X_1, int(final_n_percentile), tokenizer_from_json(final_tokenizer))
    logits = model(processed_X)
    prediction = tf.nn.softmax(logits)
    predicted_index = tf.argmax(prediction, axis=1).numpy() #[0]
    predicted_categories = label_encoder.inverse_transform(predicted_index)
    print("Logits: ", logits)
    print(predicted_categories)

    if (predicted_categories == ["entertainment", "sport", "entertainment"]).all():
        final_model.save(MODEL_STORED)
        return final_n_percentile, final_tokenizer, final_label_encoder, 1
    else:
        return 0, 0, 0, 0


def main():

    # Begin the timer to report runtime
    start_time = time.time()

    done = 0

    # Save the model's location, most effective number of words (for tokenization), and 90th percentile of input values 
    # by starting the model selection process with a call to the build model function
    while not done:
        model_ninety_percentile, model_tokenizer, model_label_encoder, done = build_model()
    # model_num_words, model_ninety_percentile, model_tokenizer = build_model()

    # Report the time taken to produce the model and the location at which it has been saved to disc
    print("Text Classification Model built and available at '%s'" % MODEL_STORED)

    # Write important information to files on disk for retrieval during fencrypt runtime
    with open(MODEL_LOCATION, "w") as file:
        file.write(MODEL_STORED)
        file.write(" ")
        # file.write(model_num_words)
        # file.write(" ")
        file.write(model_ninety_percentile)
        file.write(" ")
        file.write(TOKENIZER_STORED)
        file.write(" ")
        file.write(LABEL_ENCODER_STORED)
        file.close()

    with open(TOKENIZER_STORED, "w", encoding="utf-8") as file:
        file.write(json.dumps(model_tokenizer, ensure_ascii=False))
        file.close()

    with open(LABEL_ENCODER_STORED, 'wb') as file:
        pickle.dump(model_label_encoder, file)
        file.close()
        
    print("Process Runtime: %s minutes " % ((time.time() - start_time)/60))
    print()

if __name__ == "__main__":
    main()