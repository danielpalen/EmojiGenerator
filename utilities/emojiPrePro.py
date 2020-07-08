import imageio
#from pandas import np
import random
import numpy as np

from PIL import Image
import os


def colorChanger(x):
    if (x[3] < 20):
        return [255, 255, 255, 0]
    if (np.dot(x, [0.2126, 0.7152, 0.0722, 0]) > 250):
        return [255, 255, 255, 0]
    else:
        if ((np.dot(x, [0.2126, 0.7152, 0.0722, 0])) > 150):
            return [180, 180, 180, 0]
        else:
            return [90, 90, 90, 0]


def image_to_grey(image):
    for row in image:
        for pixel in row:
            pixel[:] = pixel_to_grey(pixel)


def pixel_to_grey(x):
    if (np.dot(x, [0.2126, 0.7152, 0.0722]) > 250):
        return [255, 255, 255]
    else:
        if ((np.dot(x, [0.2126, 0.7152, 0.0722])) > 150):
            return [180, 180, 180]
        else:
            return [90, 90, 90]


def toJpg(x):
    if (x[3] < 20):
        return [255, 255, 255, 0]
    else:
        return x


def preProcessing(image, path, file):
    imageI = Image.fromarray(image)
    imageI.rotate(random.randint(-5, 5), fillcolor='white')
    image = np.array(imageI);
    imageorg = np.copy(image)
    image = image[::2, ::2]
    image = np.repeat(np.repeat(image, 8, axis=0), 8, axis=1)
    imageorg = np.repeat(np.repeat(imageorg, 4, axis=0), 4, axis=1)
    for row in image:
        for line in row:
            line[:] = pixel_to_grey(line)

    imageio.imwrite(path + file.replace('png', 'jpg'), np.concatenate((image, imageorg), axis=1)[:, :, 0:3])


def quartering(image, path, file, is_up):
    # Choose the folders for the low or high part of an image
    if is_up:
        folders = [f'h1', f'h2', f'h3', f'h4']
    else:
        folders = [f'l1', f'l2', f'l3', f'l4']

    for f in folders:dfd
        imageio.imwrite(path + f + os.sep + file, Image.fromarray(image[:33, :33].astype(np.uint8)))