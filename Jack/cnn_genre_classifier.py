import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA_PATH = "data.json"

def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y

def prepare_datasets(test_size, validation_size):
    # load data
    X, y = load_data(DATA_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create train/validation split
    x_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train)

    # 3d array -> (130, 13, 1)
    X_train = X_train[..., np.newaxis] # 4d array -> (num_samples, 130, 13, 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):
    # create model
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D())
    # 2nc conv layer
    # 3rd conv layer
    # flatten the output and feed it into dense layer
    # output layer

if __name__ =="__main__":
    # create train, validation and test sets
    # let X = input and y = output
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # build the CNN net
    model = build_model(input_shape)

    # compile the network

    # train the CNN

    # evaluate the CNN on the test set

    # make prediction on a sample