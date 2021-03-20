# https://github.com/ianburkeixiv/mnist-reader/blob/master/mnist.py

# Author: Ian Burke
# Module: Emerging Technologies
# Mnist Reader Problem sheet

import gzip
from PIL import Image
import numpy as np
import pandas as pd


def read_labels_from_file(filename):
    with open(filename, 'rb') as f:  # use gzip to open the file in read binary mode
        magic = f.read(4)  # magic number is the first 4 bytes
        magic = int.from_bytes(magic, 'big')  # Convert bytes to integers.
        print("Magic is:", magic)  # print to console

        # the same as above but with labels
        nolab = f.read(4)
        nolab = int.from_bytes(nolab, 'big')
        print("Num of labels is:", nolab)
        # for looping through labels
        labels = [f.read(1) for i in range(nolab)]
        labels = [int.from_bytes(label, 'big') for label in labels]
    return labels


def read_images_from_file(filename):
    with open(filename, 'rb') as f:
        magic = f.read(4)
        magic = int.from_bytes(magic, 'big')
        print("Magic is:", magic)

        # Number of images in next 4 bytes
        noimg = f.read(4)
        noimg = int.from_bytes(noimg, 'big')
        print("Number of images is:", noimg)

        # Number of rows in next 4 bytes
        norow = f.read(4)
        norow = int.from_bytes(norow, 'big')
        print("Number of rows is:", norow)

        # Number of columns in next 4 bytes
        nocol = f.read(4)
        nocol = int.from_bytes(nocol, 'big')
        print("Number of cols is:", nocol)

        images = []  # create array
        # for loop
        for i in range(noimg):
            data = []
            for r in range(norow):
                for c in range(nocol):
                    data.append(int.from_bytes(f.read(1), 'big')/256)  # append the current byte for every column
            images.append(data)  # append rows for every image

    return images

def load_data(image_filename, label_filename):
    images = read_images_from_file(image_filename)
    labels = read_labels_from_file(label_filename)
    for idx, image in enumerate(images):
        image.append(labels[idx])
    cols = [str(i).zfill(3) for i in range(0, 28 * 28)]
    cols.append('cls')
    df = pd.DataFrame(images, columns=cols)
    return df

# df = load_data("data\\MNIST\\raw\\train-images-idx3-ubyte", "data\\MNIST\\raw\\train-labels-idx1-ubyte")
# print(df)
# df = load_data("data\\MNIST\\raw\\t10k-images-idx3-ubyte", "data\\MNIST\\raw\\t10k-labels-idx1-ubyte")
# train_labels = read_labels_from_file("data\\MNIST\\raw\\train-labels-idx1-ubyte")
# test_labels = read_labels_from_file("data\\MNIST\\raw\\t10k-labels-idx1-ubyte")
