import sys
import os
import time

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tkinter import PhotoImage
# from utilities import gif


class EmojiGan:
    """
    This class simplifies the creation of a generative
    adversarial network to learn generating emoji.
    """
    def __init__(self, batch_size=256, noise_dim=100, gen_lr=1e-4,
                 dis_lr=1e-4, restore_ckpt=False, examples=16, loss_func="cross_entropy"):
        self.BATCH_SIZE = batch_size
        self.NOISE_DIM = noise_dim
        self.RESTORE_CKPT = restore_ckpt
        self.EXAMPLES = examples
        self.LOSS = loss_func

        self.generator = None
        self.discriminator = None

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        if self.LOSS == "cross_entropy":
            self.generator_optimizer = tf.keras.optimizers.Adam(gen_lr)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(dis_lr)
        elif self.LOSS == "wasserstein":
            self.generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=gen_lr, clipnorm=0.1)
            self.discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=dis_lr, clipnorm=0.1)

        self.gui = None

        # ----- EXAMPLES ----- #
        # We reuse the same seed over time
        # -> Easier to visualize progress
        self.SEED = tf.random.normal([self.EXAMPLES, self.NOISE_DIM])

        # ----- OUTPUT ----- #
        if not os.path.exists(f'output/images'):
            os.mkdir(f'output/images/')

    def cross_entropy_discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def cross_entropy_generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.NOISE_DIM])

        # Initialize Gui object

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            if self.LOSS == "cross_entropy":
                gen_loss = self.cross_entropy_generator_loss(fake_output)
                disc_loss = self.cross_entropy_discriminator_loss(real_output, fake_output)

            elif self.LOSS == "wasserstein":
                gen_loss = self.wasserstein_generator_loss(fake_output)
                disc_loss = self.wasserstein_critic_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # TODO: BITTE KONTROLLIEREN, OB MAN IN TF.FUNCTION EINEN RETURN WERT HABEN KANN
        # TODO: BITT AUCH MEINE BERECHNUNG WEITER UNTEN VON DEM RUNNING AVERAGE UEBERPRUEFEN
        return gen_loss, disc_loss

    @staticmethod
    def generate_and_save_images(model, epoch, test_input, canvas_update):
        # 'Training' = False, so net runs in inference mode (batchnorm)
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            im = predictions[i].numpy()
            im = im * 127.5 + 127.5
            im = im.astype(int)
            plt.subplot(4, 4, i + 1)
            plt.imshow(im)
            plt.axis('off')
            plt.close(fig)

        plt.savefig('output/images/image_at_epoch_{:04d}.png'.format(epoch))
        if canvas_update is not None:
            print(type(canvas_update))
            print(type(canvas_update[0]))
            print(type(canvas_update[1]))
            if epoch % 10 == 0:
                img = PhotoImage(file='output/images/image_at_epoch_{:04d}.png'.format(epoch))
                canvas_update[0].after(1, canvas_update[0].itemconfig(canvas_update[1], image=img))
                print("Canvas updated.")
        # plt.show()
