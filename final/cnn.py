import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

MARSYAS_DATA_PATH = "../marsyas_data.json"
AUDIO_SET_DATA_PATH = "../audio_set_data.json"

def plot_history(history):
    fig, axis = plt.subplots(2)

    # accuracy plot
    axis[0].plot(history.history["accuracy"], label="train accuracy")
    axis[0].plot(history.history["val_accuracy"], label="test accuracy")
    axis[0].set_ylabel("Accuracy")
    axis[0].legend(loc="lower right")
    axis[0].set_title("Accuracy Eval")

    # error plot
    axis[1].plot(history.history["loss"], label="train error")
    axis[1].plot(history.history["val_loss"], label="test error")
    axis[1].set_ylabel("Error")
    axis[1].set_xlabel("Epoch")
    axis[1].legend(loc="upper right")
    axis[1].set_title("Error Eval")

    plt.show()

def predict_from_dataset(model, X, y):

    X = X[np.newaxis, ...]

    prediction = model.predict(X)

    predicted_index = np.argmax(prediction, axis=1)
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))


def main():
    with open(MARSYAS_DATA_PATH, "r") as data:
        data = json.load(data)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)

    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(10, activation='softmax'))
    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    plot_history(history)

    X = X_test[30] # random, can be any index so long as x and y have the same index
    y = y_test[30]

    predict_from_dataset(model, X, y)



if __name__ =="__main__":
    main()