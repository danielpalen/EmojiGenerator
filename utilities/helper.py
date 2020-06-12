import tensorflow as tf


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
