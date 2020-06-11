import sys
from emojigan import EmojiGan
import models
from emoji_reader import EmojiReader
from utilities import constants

# ---------- HYPERPARAMETERS ---------- #

NOISE_DIM = 100
EPOCHS = 1000
BATCH_SIZE = 256

# ---------- CREATE DATASET ----------- #

reader = EmojiReader(databases=[f'apple'], emoji_names=constants.FACE_SMILING_EMOJIS)
reader.read_images_from_sheet(pixel=32, debugging=True, png_format='RGBA')
reader.apply_preprocessing()
print(reader.images_as_np.shape)

train_dataset = reader.get_tf_dataset(batch_size=BATCH_SIZE)

# ---------- CREATE GAN ----------- #

emg = EmojiGan(
    batch_size=BATCH_SIZE, noise_dim=NOISE_DIM, gen_lr=1e-4,
    dis_lr=1e-4, restore_ckpt=False, examples=16
)
emg.generator = models.std_generator_model(
    noise_dim=100, start_shape=[8, 8, 256],
    my_layers=[[128, 5, 1], [64, 5, 2], [4, 5, 2]]
)
emg.discriminator = models.std_discriminator_model(
    input_shape=[32, 32, 4], my_layers=[[64, 5, 2, 0.3], [128, 5, 2, 0.3]]
)

# ---------- TRAINING ----------- #

emg.train(dataset=train_dataset, epochs=EPOCHS)
