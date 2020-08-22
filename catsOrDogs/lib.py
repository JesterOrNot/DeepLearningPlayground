from os import system, listdir

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


def download_data():
    system("bash scripts/getData.sh")


def create_model(x, y, epochs, batch_size):
    model = Sequential(
        [
            # Take in and flatten 200 by 200 images
            Flatten(input_dim=(200, 200)),
            # Filter out
            Dense(128, activation="relu"),
            Dense(1, activaion="sigmmoid"),
        ]
    )

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)

    return model


def get_formatted_data():
    # define location of dataset
    folder = "data/train/"
    photos, labels = list(), list()
    # enumerate files in the directory
    for file in listdir(folder):
        # determine class
        output = 0.0
        if file.startswith("cat"):
            output = 1.0
        # load image
        photo = load_img(folder + file, target_size=(200, 200))
        # convert to numpy array
        photo = img_to_array(photo)
        # store
        photos.append(photo)
        labels.append(output)
    return photos, labels

