import sys
from emojigan import EmojiGan
import models
from emoji_reader import EmojiReader
from utilities import constants
from utilities import helper
import preprocessing


class EmojiGANTraining:
    def __init__(self):
        # ---------- HYPERPARAMETERS ---------- #
        self.NOISE_DIM = 100
        self.EPOCHS = 2000
        self.BATCH_SIZE = 64
        self.GEN_LR = 2e-5
        self.DISC_LR = 2e-4

    def set_noise_dim(self, x):
        self.NOISE_DIM = x

    def set_epochs(self, x):
        self.EPOCHS = x

    def set_batch_size(self, x):
        self.BATCH_SIZE = x

    def set_gen_lr(self, x):
        self.GEN_LR = x

    def get_gen_lr(self):
        return self.GEN_LR

    def set_disc_lr(self, x):
        self.DISC_LR = x

    def training(self, image_canvas=None):
        """ Runs the DCGAN training.

        :param: image_canvas: Image canvas of the gui, to display training progress.
        :return: None
        """

        # ---------- CREATE DATASET ----------- #
        reader = EmojiReader(databases=[f'apple'], emoji_names=constants.FACE_SMILING_EMOJIS)
        images = reader.read_images_from_sheet(pixel=32, debugging=False, png_format='RGB')
        images = preprocessing.apply_std_preprocessing(images)

        train_dataset = helper.create_tf_dataset_from_np(images, batch_size=self.BATCH_SIZE)

        # ---------- CREATE GAN ----------- #

        # First create a GAN object
        emg = EmojiGan(
            batch_size=self.BATCH_SIZE, noise_dim=self.NOISE_DIM, gen_lr=2e-5,
            dis_lr=2e-4, restore_ckpt=False, examples=8, loss_func="cross_entropy"
        )
        # Add Generator
        emg.generator = models.std_generator_model(
            noise_dim=self.NOISE_DIM, start_shape=[8, 8, 256],
            my_layers=[[128, 5, 1], [64, 5, 2], [3, 5, 2]]
        )
        # Add Discriminator
        emg.discriminator = models.std_discriminator_model(
            input_shape=[32, 32, 3], my_layers=[[64, 5, 2, 0.3], [128, 5, 2, 0.3]]
        )

        # ---------- TRAINING ----------- #
        # And off you go
        emg.train(dataset=train_dataset, epochs=self.EPOCHS, image_canvas=image_canvas)


if __name__ == "__main__":
    new_training = EmojiGANTraining()
    new_training.training()

