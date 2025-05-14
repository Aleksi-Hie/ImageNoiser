import numpy as np # linear algebra
import struct
from array import array
from os.path  import join

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        arr_ = np.asarray(images, dtype=np.uint8)
        return arr_, labels
    
    def save_labels(self, filepath, labels):
        with open(filepath, 'wb') as file:
            # Write magic number and number of labels
            file.write(struct.pack(">II", 2049, len(labels)))
            # Write label data
            label_array = array("B", labels)
            label_array.tofile(file)
            
    def save_images(self, filepath, images):
        with open(filepath, 'wb') as file:
            # Write magic number, number of images, rows, and columns
            num_images = len(images)
            rows, cols = images[0].shape
            file.write(struct.pack(">IIII", 2051, num_images, rows, cols))
            # Write image data
            for img in images:
                img_array = array("B", img.flatten())
                img_array.tofile(file)

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)   