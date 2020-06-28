import sys
from emojigan import EmojiGan
import models
from emoji_reader import EmojiReader
from utilities import constants
from utilities import helper
import preprocessing

# ---------- HYPERPARAMETERS ---------- #

NOISE_DIM = 20
EPOCHS = 3000
BATCH_SIZE = 64

# ---------- CREATE DATASET ----------- #

reader = EmojiReader(databases=[f'apple'], emoji_names=constants.FACE_SMILING_EMOJIS)
images = reader.read_images_from_sheet(pixel=32, debugging=False, png_format='RGB')
images = preprocessing.apply_std_preprocessing(images)

train_dataset = helper.create_tf_dataset_from_np(images, batch_size=BATCH_SIZE)

# ---------- CREATE GAN ----------- #

# First create a GAN object
emg = EmojiGan(
    batch_size=BATCH_SIZE, noise_dim=NOISE_DIM, gen_lr=2e-5,
    dis_lr=2e-4, restore_ckpt=True, examples=8, loss_func="cross_entropy"
)
# Add Generator
emg.generator = models.std_generator_model(
    noise_dim=NOISE_DIM, start_shape=[8, 8, 256],
    my_layers=[[128, 5, 1], [64, 5, 2], [3, 5, 2]]
)
# Add Discriminator
emg.discriminator = models.std_discriminator_model(
    input_shape=[32, 32, 3], my_layers=[[64, 5, 2, 0.3], [128, 5, 2, 0.3]]
)

# ---------- TRAINING ----------- #
# And off you go
emg.train(dataset=train_dataset, epochs=EPOCHS)
