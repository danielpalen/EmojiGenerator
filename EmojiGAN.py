import os
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
                 dis_lr=1e-4, restore_ckpt=False, examples=16):
        self.BATCH_SIZE = batch_size
        self.NOISE_DIM = noise_dim
        self.RESTORE_CKPT = restore_ckpt
        self.EXAMPLES = examples

        self.generator = None
        self.discriminator = None
        self.generator_sample = None

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_optimizer = tf.keras.optimizers.Adam(gen_lr)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(dis_lr)

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

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.cross_entropy_generator_loss(fake_output)
            disc_loss = self.cross_entropy_discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def generate_and_save_images(self, epoch):
        # 'Training' = False, so net runs in inference mode (batchnorm)
        predictions = self.sample(self.SEED)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            im = predictions[i]
            plt.subplot(4, 4, i + 1)
            if len(im.shape) == 2:  # Image is grayscale
                plt.imshow(im, cmap=f'gray')
            else:
                plt.imshow(im)
            plt.axis('off')

        plt.savefig('output/images/image_at_epoch_{:04d}.png'.format(epoch))
        plt.close(fig)

    def sample(self, noise):
        predictions = self.generator(noise, training=False).numpy()
        predictions = predictions * 127.5 + 127.5 # Revert preprocessing
        predictions = predictions.astype(int)
        if predictions.shape[3] == 1:  # Image is grayscale
            predictions = np.squeeze(predictions, axis=3)
        return predictions

    def sample_and_save_single_image(self, filepath):
        noise = tf.random.normal([1, self.NOISE_DIM])  # 1 image with noise
        im = self.sample(noise)
        im = im[0]  # we only have 1 image

        self.generator_sample = im

        if len(im.shape) == 2:  # Image is grayscale
            plt.imshow(im, cmap=f'gray')
        else:
            plt.imshow(im)
        plt.axis('off')

        plt.savefig(filepath)
        plt.close()
