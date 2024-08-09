#!/usr/bin/env python3

import csv
import pandas as pd
import tensorflow as tf

def build_model():
    bbc_data = pd.read_csv("data/bbc_data.csv")
    # print(bbc_data.head())
    
    # train-test split - write here
    

    
    model_stored = "/models"
    return model_stored

def main():
    model_location = build_model()
    # model_location = "EncryptedSearch/models/"
    print("Text Classification Model built and available at %d", model_location)

if __name__ == "__main__":
    main()