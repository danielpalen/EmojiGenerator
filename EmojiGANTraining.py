import sys
import os
import models
import time
import tensorflow as tf
import numpy as np
from EmojiGAN import EmojiGan
from EmojiReader import EmojiReader
from utilities import constants
from utilities import helper
import preprocessing
import dataset_generation


class EmojiGANTraining:

    def __init__(self):
        # ---------- HYPERPARAMETERS ---------- #
        self.NOISE_DIM = 100
        self.EPOCHS = 2000
        self.BATCH_SIZE = 64
        self.GEN_LR = 2e-5
        self.DISC_LR = 2e-4
        self.PIXEL_SIZE = 32
        self.EXAMPLE_SIZE = 8
        # TODO: Seems to use checkpoint even tho this is set to false
        self.RESTORE_CHECKPOINT = False

        # ---------- OTHERS ---------- #
        self.initialization_flag = False
        self.training_flag = False
        self.train_time = None
        self.train_dataset = None
        self.emg = None
        self.checkpoint = None
        self.checkpoint_prefix = None

    def initialize(self):
        """ Runs the DCGAN training.

        :param: canvas_update: Image canvas of the gui, to display training progress.
        :return: None
        """

        color = f'gray'
        assert color in [f'RGB', f'RGBA', f'gray']

        # ---------- CREATE DATASET ----------- #
        reader = EmojiReader(databases=[f'apple'], emoji_names=constants.FACE_EMOJIS_DCGAN_TRAINING)
        images = reader.read_images_from_sheet(pixel=self.PIXEL_SIZE, debugging=False, png_format=color)

        images = preprocessing.apply_std_preprocessing(images)

        self.train_dataset = helper.create_tf_dataset_from_np(images, batch_size=self.BATCH_SIZE)

        # ---------- CREATE GAN ----------- #

        # First create a GAN object
        self.emg = EmojiGan(
            batch_size=self.BATCH_SIZE, noise_dim=self.NOISE_DIM, gen_lr=self.GEN_LR,
            dis_lr=self.DISC_LR, restore_ckpt=self.RESTORE_CHECKPOINT, examples=self.EXAMPLE_SIZE
        )
        if color == f'RGB':
            # Add Generator
            self.emg.generator = models.std_generator_model(
                noise_dim=self.NOISE_DIM, start_shape=[8, 8, 256],
                my_layers=[[128, 5, 1], [64, 5, 2], [3, 5, 2]]
            )
            # Add Discriminator
            self.emg.discriminator = models.std_discriminator_model(
                input_shape=[32, 32, 3], my_layers=[[64, 5, 2, 0.3], [128, 5, 2, 0.3]]
            )
        elif color == f'gray':
            # Add Generator
            self.emg.generator = models.std_generator_model(
                noise_dim=self.NOISE_DIM, start_shape=[8, 8, 256],
                my_layers=[[128, 5, 1], [64, 5, 2], [1, 5, 2]]
            )
            # Add Discriminator
            self.emg.discriminator = models.std_discriminator_model(
                input_shape=[32, 32, 1], my_layers=[[64, 5, 2, 0.3], [128, 5, 2, 0.3]]
            )

        else:
            NotImplementedError()

        # ----- CHECKS ----- #
        # Generator and discirminator have been set
        if not self.emg.generator or not self.emg.discriminator:
            raise RuntimeError(f'The generator and discriminator have to be set before training.')

        # ----- CKPT CONFIG ----- #
        if not os.path.exists(f'output/checkpoints/'):
            os.mkdir(f'output/checkpoints/')
        checkpoint_dir = 'output/checkpoints'
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.emg.generator_optimizer,
            discriminator_optimizer=self.emg.discriminator_optimizer,
            generator=self.emg.generator,
            discriminator=self.emg.discriminator)

        if self.emg.RESTORE_CKPT:
            self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        self.train_time = time.time()

        # After initialization, this variable is set to True, so
        # the while loop of the GUI that waits for the user input
        # can be exited.
        self.initialization_flag = True

    def sample(self):
        """
            Generates a sample from the DCGAN architecture using the last checkpoint from output/checkpoints.

        """
        self.RESTORE_CHECKPOINT = True
        self.initialize()
        noise = tf.random.normal([1, self.NOISE_DIM])
        sample = self.emg.generator(noise, training=False).numpy()
        self.RESTORE_CHECKPOINT = False
        self.initialization_flag = False
        return sample


