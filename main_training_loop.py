import time
from Gui import Gui
from EmojiGANTraining import EmojiGANTraining
import tensorflow as tf

# Avoid an error occuring on some GPU's when running tensorflow
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu_devices[0], True)

use_gui = True

training_instance = EmojiGANTraining()
gui_instance = Gui(training_instance)

if use_gui:
    gui_instance.build_gui()
else:
    training_instance.initialize()

# Main gui loop
while True:
    
    # Sleep while training instance is not initialized and not ready for training
    while ((not training_instance.initialization_flag) or (not training_instance.training_flag)) and use_gui:
        time.sleep(0.01)
        gui_instance.root.update()

    # ----- MAIN TRAINING LOOP ----- #
    for epoch in range(training_instance.EPOCHS + 1):
        start = time.time()

        gen_loss_avg = 0
        disc_loss_avg = 0
        batch_counter = 0

        for image_batch in training_instance.train_dataset:
            gen_loss, disc_loss = training_instance.emg.train_step(image_batch)

            gen_loss_avg += gen_loss
            disc_loss_avg += disc_loss
            batch_counter += 1

        gen_loss_avg = gen_loss_avg / batch_counter
        disc_loss_avg = disc_loss_avg / batch_counter

        # Produce images for the GIF as we go
        training_instance.emg.generate_and_save_images(training_instance.emg.generator, epoch + 1,
                                                       training_instance.emg.SEED)

        # Save the model every 15 epochs
        if epoch % 15 == 0:
            training_instance.checkpoint.save(file_prefix=training_instance.checkpoint_prefix)

        print(f'Epoch {format(epoch + 1, "4")}, Time {format(time.time() - start, ".2f")} sec, ' +
              f'Gen loss: {format(gen_loss_avg, ".4f")}, '
              f'Disc loss: {format(disc_loss_avg, ".4f")}')

        # Update image canvas in trianing tab (tab 2) after each episode of training
        if use_gui:
            gui_instance.image_canvas_update_1(path='output/images/image_at_epoch_{:04d}.png'.format(epoch),
                                               progress_text_update=f"{epoch}/{training_instance.EPOCHS}")

        # Break training loop if training button is pressed again
        if use_gui and training_instance.training_flag is False:
            break

    # Generate after the final epoch
    training_instance.emg.generate_and_save_images(training_instance.emg.generator, training_instance.EPOCHS,
                                                   training_instance.emg.SEED)

    print(f'\nTRAINING FINISHED (Time: {format(time.time() - training_instance.train_time, ".2f")} sec)')

    # gif.create_gif(f'output/images/image*.png', f'output/emojigan.gif')
