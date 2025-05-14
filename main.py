import cv2 as cv
import numpy as np
import os
from env import *
from mnistdataloader import MnistDataloader
def read_files(Path):
    for file_name in os.listdir(Path):
        file_path = os.path.join(Path, file_name)
        if os.path.isfile(file_path):
            yield file_path

def get_path():
    return input("Give path to dataset: ")

def GaussianNoise(image):
    return cv.GaussianBlur(image, (5,5),0)

def save_image(image, Path):
    cv.imwrite(os.path.join(Path, "noisy"+Path), image)

def add_noise(images, noise):
    noisyset = []
    for i, image in enumerate(images):
        noisy = noise(image)
        noisyset.append(noisy)
    return noisyset



def main():

    save_training = save_location+"noisy_training"
    save_test = save_location+"noisy_test"

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    x_train_noisy = add_noise(x_train, GaussianNoise)
    x_test_noisy = add_noise(x_test, GaussianNoise)

    

    mnist_dataloader.save_images(save_training, x_train_noisy)
    mnist_dataloader.save_images(save_test, x_test_noisy)



if __name__ == "__main__":
    main()