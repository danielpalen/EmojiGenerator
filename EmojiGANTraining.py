import os
import models
import time
import tensorflow as tf
from emojigan import EmojiGan
from emoji_reader import EmojiReader
from utilities import constants
from utilities import helper
import preprocessing


class EmojiGANTraining:

    def __init__(self, canvas_update=None):
        # ---------- HYPERPARAMETERS ---------- #
        self.NOISE_DIM = 100
        self.EPOCHS = 2000
        self.BATCH_SIZE = 64
        self.GEN_LR = 2e-5
        self.DISC_LR = 2e-4
        self.RESTORE_CHECKPOINT = False

        # ---------- OTHERS ---------- #
        self.initialization_flag = False
        self.train_time = None
        self.train_dataset = None
        self.emg = None
        self.canvas_update = canvas_update
        self.checkpoint = None
        self.checkpoint_prefix = None

    def initialize(self):
        """ Runs the DCGAN training.

        :param: canvas_update: Image canvas of the gui, to display training progress.
        :return: None
        """

        # ---------- CREATE DATASET ----------- #
        reader = EmojiReader(databases=[f'apple'], emoji_names=constants.FACE_SMILING_EMOJIS)
        images = reader.read_images_from_sheet(pixel=32, debugging=False, png_format='RGB')
        images = preprocessing.apply_std_preprocessing(images)

        self.train_dataset = helper.create_tf_dataset_from_np(images, batch_size=self.BATCH_SIZE)

        # ---------- CREATE GAN ----------- #

        # First create a GAN object
        self.emg = EmojiGan(
            batch_size=self.BATCH_SIZE, noise_dim=self.NOISE_DIM, gen_lr=2e-5,
            dis_lr=2e-4, restore_ckpt=self.RESTORE_CHECKPOINT, examples=8, loss_func="cross_entropy"
        )
        # Add Generator
        self.emg.generator = models.std_generator_model(
            noise_dim=self.NOISE_DIM, start_shape=[8, 8, 256],
            my_layers=[[128, 5, 1], [64, 5, 2], [3, 5, 2]]
        )
        # Add Discriminator
        self.emg.discriminator = models.std_discriminator_model(
            input_shape=[32, 32, 3], my_layers=[[64, 5, 2, 0.3], [128, 5, 2, 0.3]]
        )

        # ----- CHECKS ----- #
        # Generator and discirminator have been set
        if not self.emg.generator or not self.emg.discriminator:
            raise RuntimeError(f'The generator and discriminator have to be set before training.')

        # TODO: check if dataset has tf.float32 datatype
        # raise TypeError(f'Please convert dataset to float before preprocessing.')

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

        self.initialization_flag = True






