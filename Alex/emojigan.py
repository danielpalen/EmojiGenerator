
from read_json import read_json
import matplotlib.pyplot as plt
import imageio
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
import sys

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

data = read_json()
emojis = []

# Extract only smileys and emotions
for i in range(len(data)):
    if data[i][f'category'] == f'Smileys & Emotion':
        emojis.append(data[i])

# ---------- Extracting the images ----------- #

im_path = f'../emoji-data/sheet_apple_32.png'
pixel = 32

# Remember this!!! This is how to read and write png preserving quality
img = imageio.imread(im_path)
img = np.asarray(img, dtype='float')
imageio.imwrite(f'test.png', img)

print(len(img))

imgs = []

for i in range(len(emojis)):
    em = emojis[i]
    x = em[f'sheet_x']
    y = em[f'sheet_y']

    gaps_x = 1 + x*2
    gaps_y = 1 + y*2

    start_x = gaps_x + pixel * x
    start_y = gaps_y + pixel * y
    imgs.append(img[start_x:start_x+pixel, start_y:start_y+pixel])

imgs = np.asarray(imgs)
print(imgs.shape)
print(type(imgs))
#imgs = imgs.astype('float32')

print(imgs[1])

plt.imshow(imgs[1])
plt.show()

sys.exit()

# ---------- CREATE MODELS ----------- #

def make_generator_model():
    """
    Conv2DTranspose for upsampling
    Starting with Dense layer that takes the noise seed as input
    LeakyReLU activation, except last layer
    """
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 32, 32, 1)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[32, 32, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


generator = make_generator_model()
discriminator = make_discriminator_model()
