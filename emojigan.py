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

    def wasserstein_critic_loss(self, real_output, fake_output):
        return tf.math.reduce_mean(real_output) - tf.math.reduce_mean(fake_output)

    def cross_entropy_generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def wasserstein_generator_loss(self, fake_output):
        return tf.reduce_mean(fake_output)

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.NOISE_DIM])

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

    def train(self, dataset, epochs):

        # ----- CHECKS ----- #
        # Generator and discirminator have been set
        if not self.generator or not self.discriminator:
            raise RuntimeError(f'The generator and discriminator have to be set before training.')

        # TODO: check if dataset has tf.float32 datatype
        # raise TypeError(f'Please convert dataset to float before preprocessing.')

        # ----- CKPT CONFIG ----- #
        if not os.path.exists(f'output/checkpoints/'):
            os.mkdir(f'output/checkpoints/')
        checkpoint_dir = 'output/checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator)

        if self.RESTORE_CKPT:
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        train_time = time.time()

        for epoch in range(epochs):
            start = time.time()

            gen_loss_avg = 0
            disc_loss_avg = 0
            batch_counter = 0

            for image_batch in dataset:
                gen_loss, disc_loss = self.train_step(image_batch)

                gen_loss_avg += gen_loss
                disc_loss_avg += disc_loss
                batch_counter += 1

            gen_loss_avg = gen_loss_avg / batch_counter
            disc_loss_avg = disc_loss_avg / batch_counter

            # Produce images for the GIF as we go
            self.generate_and_save_images(self.generator, epoch + 1, self.SEED)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            print(f'Epoch {format(epoch+1, "4")}, Time {format(time.time()-start, ".2f")} sec, ' +
                  f'Gen loss: {format(gen_loss_avg, ".4f")}, '
                  f'Disc loss: {format(disc_loss_avg, ".4f")}')

        # Generate after the final epoch
        self.generate_and_save_images(self.generator, epochs, self.SEED)

        print(f'\nTRAINING FINISHED (Time: {format(time.time() - train_time, ".2f")} sec)')

        # gif.create_gif(f'output/images/image*.png', f'output/emojigan.gif')


    @staticmethod
    def generate_and_save_images(model, epoch, test_input):
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

        image_canvas = None
        image_on_canvas = None
        plt.savefig('output/images/image_at_epoch_{:04d}.png'.format(epoch))
        if (image_canvas is not None) and (image_on_canvas is not None):
            if epoch % 10 == 0:
                img = PhotoImage(file='output/images/image_at_epoch_{:04d}.png'.format(epoch))
                image_canvas.itemconfig(image_on_canvas, image=img)
                print("Image canvas updated.")
        # plt.show()
