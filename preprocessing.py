import numpy as np
import tensorflow as tf


def apply_std_preprocessing(images):
    """
    Convert a numpy image array that contains channels ranging
    from 0 to 255 to float and let them fall around 0 into [-1, 1].
    :return: numpy array of images.
    """
    images = np.asarray(images, np.float32)
    images = (images - 127.5) / 127.5

    print(f'PREPROCESSING:')
    print(f'- {np.average(images, axis=(0, 1, 2))} should be around 0')
    print(f'- {np.min(images, axis=(0, 1, 2))} should be around -1')
    print(f'- {np.max(images, axis=(0, 1, 2))} should be around 1\n')

    return images