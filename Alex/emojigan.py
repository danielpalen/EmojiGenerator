
from Alex.read_json import read_json
import matplotlib.pyplot as plt
import imageio
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
import sys
import os
import time
from PIL import Image
from emoji_reader import EmojiReader
"""
    | Fields | Description |
    | ------ | ----------- |
    | `name` | The offical Unicode name, in SHOUTY UPPERCASE. |
    | `category` | The offical Unicode name, in SHOUTY UPPERCASE. |
    | `unified` | The Unicode codepoint, as 4-5 hex digits. Where an emoji needs 2 or more codepoints, they are specified like 1F1EA-1F1F8. For emoji that need to specifiy a variation selector (-FE0F), that is included here. |
    | `non_qualified` | For emoji that also have usage without a variation selector, that version is included here (otherwise is null). |
    | `docomo`, `au`,<br>`softbank`, `google` | The legacy Unicode codepoints used by various mobile vendors. |
    | `image` | The name of the image file. |
    | `sheet_x`, `sheet_y` | The position of the image in the spritesheets. |
    | `short_name` | The commonly-agreed upon short name for the image, as supported in campfire, github etc via the :colon-syntax: |
    | `short_names` | An array of all the known short names. |
    | `text` | An ASCII version of the emoji (e.g. `:)`), or null where none exists. |
    | `texts` | An array of ASCII emoji that should convert into this emoji. Each ASCII emoji will only appear against a single emoji entry. |
    | `has_img_*` | A flag for whether the given image set has an image (named by the image prop) available. |
    | `added_id` | Emoji version in which this codepoint/sequence was added (previously Unicode version). |
    | `skin_variations` | For emoji with multiple skin tone variations, a list of alternative glyphs, keyed by the skin tone. For emoji that support multiple skin tones within a single emoji, each skin tone is separated by a dash character. |
    | `obsoletes`, `obsoleted_by` | Emoji that are no longer used, in preference of gendered versions. |

"""
"""
    Categories:
    {'Travel & Places', 'Objects', 'Animals & Nature', 'Skin Tones', 'Activities', 
    'Symbols', 'People & Body', 'Food & Drink', 'Flags', 'Smileys & Emotion'}
"""

reader = EmojiReader(categories=['Smileys & Emotion'])
emojis = reader.selected_meta_data



# ---------- PREPROCESSING ----------- #

# Alpha channel has range [0, 255]

imgs = np.asarray(imgs)
imgs = imgs.astype(float)
imgs = (imgs - 127.5) / 127.5
print(f'{np.average(imgs, axis=(0, 1, 2))} should be around 0')
print(f'{np.min(imgs, axis=(0, 1, 2))} should be around -1')
print(f'{np.max(imgs, axis=(0, 1, 2))} should be around 1')


# ---------- CREATE DATASET ----------- #

BUFFER_SIZE = 60000
BATCH_SIZE = 32

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(imgs).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

print(train_dataset.element_spec)

# ---------- CREATE MODELS ----------- #


def make_generator_model():
    """
    Conv2DTranspose for upsampling
    Starting with Dense layer that takes the noise seed as input
    LeakyReLU activation, except last layer
    """
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*1024, use_bias=False, input_shape=(10,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 1024)))
    assert model.output_shape == (None, 8, 8, 1024) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(521, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 521)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 32, 32, 3)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[32, 32, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


generator = make_generator_model()
discriminator = make_discriminator_model()


# ---------------- LOSS AND OPTIMIZERS --------------- #

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# ---------------- SAVE CHECKPOINTS --------------- #


checkpoint_dir = '../output/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# ------------------ TRAINING LOOP ---------------- #

# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

EPOCHS = 1000
noise_dim = 10
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as we go
    generate_and_save_images(generator, epoch + 1, seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      im = predictions[i].numpy()
      im = im * 127.5 + 127.5
      im = im.astype(int)
      plt.subplot(4, 4, i + 1)
      plt.imshow(im)
      plt.axis('off')

  plt.savefig('./images/image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()


train(train_dataset, EPOCHS)

print(f'FINISHED')
