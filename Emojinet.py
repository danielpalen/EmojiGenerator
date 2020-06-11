import os
import time

import tensorflow as tf
import matplotlib.pyplot as plt


class EmojiGan:
    """
    This class simplifies the creation of a generative
    adversarial network to learn generating emoji.
    """
    def __init__(self, batch_size, noise_dim, restore_ckpt=False, examples=16):
        self.BATCH_SIZE = batch_size
        self.NOISE_DIM = noise_dim
        self.RESTORE_CKPT = restore_ckpt
        self.EXAMPLES = examples

        self.generator = None
        self.discriminator = None

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        # ----- EXAMPLES ----- #
        # We reuse the same seed over time
        # -> Easier to visualize progress
        self.SEED = tf.random.normal([self.EXAMPLES, self.NOISE_DIM])

        # ----- CHECKPOINTS ----- #
        if not os.path.exists(f'output/checkpoints/'):
            os.mkdir(f'output/checkpoints/')
        self.checkpoint_prefix = f'output/checkpoints/ckpt'
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator)

        # ----- OUTPUT ----- #
        if not os.path.exists(f'output/images'):
            os.mkdir(f'output/images/')

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
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

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, dataset, epochs):

        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                self.train_step(image_batch)

            # Produce images for the GIF as we go
            self.generate_and_save_images(self.generator, epoch + 1, self.SEED)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
              self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # Generate after the final epoch
        self.generate_and_save_images(self.generator, epochs, self.SEED)

        print(f'TRAINING FINISHED')

    @staticmethod
    def generate_and_save_images(self, model, epoch, test_input):
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

        plt.savefig('output/images/image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()