import tensorflow as tf

from tensorflow.keras import layers


def std_generator_model(noise_dim, start_shape, my_layers):
    """
    TODO: Description.
    Conv2DTranspose for upsampling
    Starting with Dense layer that takes the noise seed as input
    LeakyReLU activation, except last layer
    """

    assert start_shape[0] == start_shape[1]

    layer_width = start_shape[0]

    model = tf.keras.Sequential()
    model.add(layers.Dense(start_shape[0] * start_shape[1] * start_shape[2],
                           use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((start_shape[0], start_shape[0], start_shape[2])))
    assert model.output_shape == (None, start_shape[0], start_shape[0], start_shape[2]) # Note: None is the batch size

    for i, ml in enumerate(my_layers):

        layer_width = layer_width * ml[2]

        model.add(layers.Conv2DTranspose(ml[0], (ml[1], ml[1]), strides=(ml[2], ml[2]), padding='same', use_bias=False))
        assert model.output_shape == (None, layer_width, layer_width, ml[0])

        if i+1 == len(my_layers):
            model.add(layers.Activation(f'tanh'))
        else:
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

    print(f'GENERATOR NETWORK MODEL')
    print(model.summary())

    return model


def std_discriminator_model(input_shape=None, my_layers=None, std_dropout=0.3):
    """
    TODO: Description
    :param input_shape:
    :param my_layers:
    :param std_dropout:
    :return:
    """
    if input_shape is None:
        input_shape = [32, 32, 3]
    if my_layers is None:
        my_layers = [[64, 5, 2, 0.3], [128, 5, 2, 0.3]]

    model = tf.keras.Sequential()

    for i, ml in enumerate(my_layers):
        if i == 0:
            model.add(layers.Conv2D(ml[0], (ml[1], ml[1]), strides=(ml[2], ml[2]),
                                    padding='same', input_shape=input_shape))
        else:
            model.add(layers.Conv2D(ml[0], (ml[1], ml[1]), strides=(ml[2], ml[2]), padding='same'))
        model.add(layers.LeakyReLU())

        if len(ml) == 4:
            model.add(layers.Dropout(ml[3]))
        else:
            model.add(layers.Dropout(std_dropout))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    print(f'DISCRIMINATOR NETWORK MODEL')
    print(model.summary())

    return model
