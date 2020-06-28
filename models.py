import tensorflow as tf

from tensorflow.keras import layers


def std_generator_model(noise_dim, start_shape, my_layers):
    """
    This function generates the generator network of a standard
    deep convolutional GAN. The implementation follows
    https://www.tensorflow.org/tutorials/generative/dcgan, which
    is based on the paper https://arxiv.org/pdf/1511.06434.pdf

    Properties are:
    - Input noise is connected to a dense layer
    - The dense layer is reshaped into a desired start_shape
    - Each following layer uses upsampling (the transpose of
      convolutions)
    - After each layer we use batch normalization
    - and leaky ReLU activation, except last layer (tanh)

    :param noise_dim: int, the dimension of the input noise vector
    :param start_shape: list or tuple of three numbers, determining
        the shape of the first neuron block. First two numbers need
        to be the same and determine the width and height, whereas
        the third entry determines the depth = number of filters.
        Example: [8, 8, 32].
    :param my_layers: list of lists, where each inner list contains
        three numbers. First entry is the number of filters (depth)
        in the upsampling layer, the second is the filter size
        (so 5 would result in a filter size of 5x5) and the third
        number is the stride. If a stride of > 1 is used, the output
        will grow, because we use upsampling layers.
        Example: [[128, 5, 1], [64, 5, 2], [3, 5, 2]]
    """

    # We want symmetric start shape
    assert start_shape[0] == start_shape[1]

    # Saves current width of the network
    net_width = start_shape[0]

    # Connect input to dense layer
    model = tf.keras.Sequential()
    model.add(layers.Dense(start_shape[0] * start_shape[1] * start_shape[2],
                           use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Reshape dense layer into start_shape (no BN and activation)
    model.add(layers.Reshape((start_shape[0], start_shape[0], start_shape[2])))
    assert model.output_shape == (None, start_shape[0], start_shape[0], start_shape[2]) # Note: None is the batch size

    for i, ml in enumerate(my_layers):

        # Net width only increases if the stride is > 1
        net_width = net_width * ml[2]

        model.add(layers.Conv2DTranspose(ml[0], (ml[1], ml[1]), strides=(ml[2], ml[2]), padding='same', use_bias=False))
        assert model.output_shape == (None, net_width, net_width, ml[0])

        if i+1 == len(my_layers):
            model.add(layers.Activation(f'tanh'))
        else:
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

    print(f'GENERATOR NETWORK MODEL')
    print(model.summary())

    return model


def std_discriminator_model(input_shape, my_layers, std_dropout=0.3):
    """
    This function generates the discriminator network of a standard
    deep convolutional GAN. The implementation follows
    https://www.tensorflow.org/tutorials/generative/dcgan, which
    is based on the paper https://arxiv.org/pdf/1511.06434.pdf

    Properties are:
    - Stacked convolutional layer
    - Followed by LeakyReLU activation and dropout
    - At the end, the layer is flattened and fully connected to
      a single output.

    :param input_shape: list or tuple of three numbers. The first
        two numbers determin the width and height of the input image.
        The third number determines the number of channels.
        Example: [32, 32, 3]
    :param my_layers: list of lists, where each inner list contains
        three or four numbers. First entry is the number of filters
        (depth) in the convolutional layer, the second is the filter
        size (so 5 would result in a filter size of 5x5) and the third
        number is the stride. If a stride of > 1 is used, the output
        will shrink, because we use convolutional layers. The fourth
        entry can be set if we want to specify for this layer a
        different dropout rate than given by std_dropout.
        Example: [[64, 5, 2, 0.3], [128, 5, 2, 0.3]]
    :param std_dropout: float between 0 and 1. Determines the dropout
        rates for all layers, when non is given explicitly.
    """

    assert 0 <= std_dropout <= 1

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

    # TODO: Should we not use a sigmoid for the output for binary classification?

    print(f'DISCRIMINATOR NETWORK MODEL')
    print(model.summary())

    return model
