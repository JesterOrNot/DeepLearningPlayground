from os import system, listdir, makedirs
from shutil import copyfile
from random import seed, random

from keras.layers import Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator


def download_data():
    system("bash scripts/getData.sh")


def create_model(x, y, epochs, batch_size):
    print("Defining Model...")
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)),
            Flatten(),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            MaxPooling2D((2, 2))
        ]
    )
    print("Compiling Model...")
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    print("Fitting Model...")
    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)

    return model


def prep_data():
    # create directories
    dataset_home = "dataset_dogs_vs_cats/"
    subdirs = ["train/", "test/"]
    for subdir in subdirs:
        # create label subdirectories
        labeldirs = ["dogs/", "cats/"]
        for labldir in labeldirs:
            newdir = dataset_home + subdir + labldir
            makedirs(newdir, exist_ok=True)
    # seed random number generator
    seed(1)
    # define ratio of pictures to copy
    val_ratio = 0.25
    # copy training dataset images into subdirectories
    src_directory = "data/train/"
    for file in listdir(src_directory):
        src = src_directory + file
        dst_dir = "train/"
        if random() < val_ratio:
            dst_dir = "test/"
        if file.startswith("cat"):
            dst = dataset_home + dst_dir + "cats/" + file
            copyfile(src, dst)
        elif file.startswith("dog"):
            dst = dataset_home + dst_dir + "dogs/" + file
            copyfile(src, dst)

def remove_old_data():
    system("rm -rf data")

def get_model_data():
  data_generator = ImageDataGenerator(rescale=1.0/255.0)
  # prepare iterators
  train_data = data_generator.flow_from_directory('dataset_dogs_vs_cats/train/',
    class_mode='binary', batch_size=64, target_size=(200, 200))
  test_data = data_generator.flow_from_directory('dataset_dogs_vs_cats/test/',
    class_mode='binary', batch_size=64, target_size=(200, 200))
  return train_data, test_data
