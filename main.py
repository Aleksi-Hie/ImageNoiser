import cv2 as cv
import numpy as np
import os

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
    cv.imwrite(os.path.join(Path, "noisy"+Path))

def add_noise(Path, noise, callback):
    for image_path in read_files(Path):
        image = cv.imread(image_path)
        noise(image)
        callback(image, image_path)



def main():
    path =get_path()
    add_noise(path, GaussianNoise, save_image)

if __name__ == "__main__":
    main()