import imageio
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def create_tf_dataset_from_np(images, batch_size):
    """
    Create a tensorflow dataset which is shuffled and divided
    into batches. Make sure each entry in the image array
    represents an image.

    :param images: a numpy array of images. 4 dimensional.
    :param batch_size: the desired batch size
    :return: a tensorflow Dataset
    """

    assert len(images.shape) == 4

    buffer_size = 60000

    # Batch and shuffle the data
    tf_dataset = tf.data.Dataset.from_tensor_slices(images)
    tf_dataset = tf_dataset.shuffle(buffer_size).batch(batch_size)

    return tf_dataset


BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


def load(image_file):
    image = image_file
    # image = tf.io.read_file(image_file)
    # image = tf.image.decode_jpeg(image_file)

    input_image = tf.cast(image, tf.float32)
    return input_image, input_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


def load_image(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                     IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def predict_image_pix2pix(image, model_path):
    generator = tf.keras.models.load_model(model_path,compile=False)

    images = []

    width = image.shape[0]
    if (width == 32):
        image = np.repeat(np.repeat(image, 8, axis=0), 8, axis=1)

    images.append(image)

    train_dataset = tf.data.Dataset.from_tensor_slices(images)
    train_dataset = train_dataset.map(load_image)
    train_dataset = train_dataset.batch(1)

    for inp, tar in train_dataset.take(1):
        prediction = generator(inp, training=True)
        return prediction[0]
