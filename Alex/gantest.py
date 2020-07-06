import sys
from EmojiGAN import EmojiGan
import models
from EmojiReader import EmojiReader
from utilities import constants
from utilities import helper
import preprocessing

# ---------- HYPERPARAMETERS ---------- #

NOISE_DIM = 100
EPOCHS = 500
BATCH_SIZE = 256

# ---------- CREATE DATASET ----------- #

reader = EmojiReader(databases=[f'apple'], emoji_names=constants.FACE_SMILING_EMOJIS)
images = reader.read_images_from_sheet(pixel=32, debugging=False, png_format='RGB')
images = preprocessing.apply_std_preprocessing(images)

train_dataset = helper.create_tf_dataset_from_np(images, batch_size=BATCH_SIZE)

# ---------- CREATE GAN ----------- #

# First create a GAN object
emg = EmojiGan(
    batch_size=BATCH_SIZE, noise_dim=NOISE_DIM, gen_lr=1e-4,
    dis_lr=1e-4, restore_ckpt=False, examples=16
)
# Add Generator
emg.generator = models.std_generator_model(
    noise_dim=100, start_shape=[8, 8, 256],
    my_layers=[[128, 5, 1], [64, 5, 2], [3, 5, 2]]
)
# Add Discriminator
emg.discriminator = models.std_discriminator_model(
    input_shape=[32, 32, 3], my_layers=[[64, 5, 2, 0.3], [128, 5, 2, 0.3]]
)

# ---------- TRAINING ----------- #
# And off you go
emg.train(dataset=train_dataset, epochs=EPOCHS)
