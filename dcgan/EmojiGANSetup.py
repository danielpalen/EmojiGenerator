import os
from dcgan import models, preprocessing
import time
import tensorflow as tf
from dcgan.EmojiGAN import EmojiGan
from EmojiReader import EmojiReader
from utilities import constants
from utilities import helper
import shutil


class EmojiGANTraining:
    """
        Class for handling the training set generation
        and the set up of our custom DCGAN networks,
        which we subsequently train with the
        methods provided in EmojiGAN.py
    """
    def __init__(self):
        # ---------- HYPERPARAMETERS ---------- #
        self.NOISE_DIM = 100
        self.EPOCHS = 2000
        self.BATCH_SIZE = 64
        self.GEN_LR = 2e-5
        self.DISC_LR = 2e-4
        self.PIXEL_SIZE = 32
        self.EXAMPLE_SIZE = 8
        self.COLORSPACE = f'RGB'
        self.RESTORE_CHECKPOINT = True

        # ---------- OTHERS ---------- #
        self.initialization_flag = False
        self.training_flag = False
        self.train_time = None
        self.train_dataset = None
        self.emg = None
        self.checkpoint = None
        self.checkpoint_prefix = None

    def initialize(self):
        """
            Read and select emojis with EmojiReader.py from hard
            drive. Apply preprocessing from preprocessing.py
            to the images and create a training dataset.

            Then, create an EmojiGAN instance with parameters
            either read from __init__ of this class (no GUI)
            or entered via GUI.

            Finally, we create a generator and discriminator
            network using the modular classes in models.py
            and pass them to the EmojiGAN instance.

            After that training can be started (GUI: hitting
            training button; No GUI: automatically).
        """

        assert self.COLORSPACE in [f'RGB', f'RGBA', f'gray']

        # ---------- CREATE DATASET ----------- #
        reader = EmojiReader(databases=[f'apple'], emoji_names=constants.FACE_SMILING_EMOJIS)
        images = reader.read_images_from_sheet(pixel=self.PIXEL_SIZE, debugging=False, png_format=self.COLORSPACE)

        images = preprocessing.apply_std_preprocessing(images)

        self.train_dataset = helper.create_tf_dataset_from_np(images, batch_size=self.BATCH_SIZE)

        # ---------- CREATE GAN ----------- #

        # First create a GAN object
        self.emg = EmojiGan(
            batch_size=self.BATCH_SIZE, noise_dim=self.NOISE_DIM, gen_lr=self.GEN_LR,
            dis_lr=self.DISC_LR, restore_ckpt=self.RESTORE_CHECKPOINT, examples=self.EXAMPLE_SIZE
        )
        if self.COLORSPACE == f'RGB':
            # Add Generator
            self.emg.generator = models.std_generator_model(
                noise_dim=self.NOISE_DIM, start_shape=[8, 8, 256],
                my_layers=[[128, 5, 1], [64, 5, 2], [3, 5, 2]]
            )
            # Add Discriminator
            self.emg.discriminator = models.std_discriminator_model(
                input_shape=[32, 32, 3], my_layers=[[64, 5, 2, 0.3], [128, 5, 2, 0.3]]
            )
        elif self.COLORSPACE == f'gray':
            # Add Generator
            self.emg.generator = models.std_generator_model(
                noise_dim=self.NOISE_DIM, start_shape=[4, 4, 1024],
                my_layers=[[512, 5, 2], [256, 5, 1], [128, 5, 2], [1, 5, 2]]
            )
            # Add Discriminator
            self.emg.discriminator = models.std_discriminator_model(
                input_shape=[32, 32, 1], my_layers=[[64, 5, 2, 0.3], [128, 5, 2, 0.3]]
            )
        else:
            NotImplementedError()

        # ----- CHECKS ----- #
        # Generator and discriminator have been set
        if not self.emg.generator or not self.emg.discriminator:
            raise RuntimeError(f'The generator and discriminator have to be set before training.')

        # ----- CKPT CONFIG ----- #
        # Convert string (from GUI) to bool
        if type(self.emg.RESTORE_CKPT) == str:
            if self.emg.RESTORE_CKPT.lower() in [f'true', f'1', f'y', f'yes']:
                self.emg.RESTORE_CKPT = True
            else:
                self.emg.RESTORE_CKPT = False

        checkpoint_dir = f'output/checkpoints'
        if not self.emg.RESTORE_CKPT:
            shutil.rmtree(checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.emg.generator_optimizer,
            discriminator_optimizer=self.emg.discriminator_optimizer,
            generator=self.emg.generator,
            discriminator=self.emg.discriminator)

        if self.emg.RESTORE_CKPT:
            self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            print(f'LAST CHECKPOINT LOADED (from output/checkpoint)')

        self.train_time = time.time()

        # After initialization, this variable is set to True, so
        # the while loop of the GUI that waits for the user input
        # can be exited and the training is started.
        self.initialization_flag = True

        print(f'\nInitialization Done!')

    def load_ckpt_and_sample_img(self, filepath):
        """
            Generates a sample from the DCGAN architecture using
            the last checkpoint from output/checkpoints.
        """
        self.RESTORE_CHECKPOINT = True
        self.initialize()
        self.emg.sample_and_save_single_image(filepath)
        self.RESTORE_CHECKPOINT = False
        self.initialization_flag = False
        return True


