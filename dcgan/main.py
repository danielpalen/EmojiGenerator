import time
import os
from dcgan.Gui import Gui
from dcgan.EmojiGANSetup import EmojiGANTraining
from utilities import gif

# --------------- SET DEVICES --------------- #
# Decide, whether you want to train on CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Uncomment if your GPU needs the command
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu_devices[0], True)

# --------------- GUI SETTINGS --------------- #
use_gui = True

training_instance = EmojiGANTraining()
gui_instance = Gui(training_instance)

if use_gui:
    gui_instance.build_gui()
else:
    training_instance.initialize()

# --------------- CREATE OUTPUT FOLDER --------------- #
if not os.path.isdir(f'output'):
    os.mkdir(f'output')

# --------------- GUI LOOP --------------- #
while True:

    # GUI: Stop training after one training cycle, until button in GUI is pressed again
    training_instance.training_flag = False

    # GUI: Sleep while training instance is not initialized and not ready for training
    while (not training_instance.initialization_flag or not training_instance.training_flag) and use_gui:
        time.sleep(0.01)
        gui_instance.root.update()

    # -------- MAIN TRAINING LOOP -------- #
    for epoch in range(training_instance.EPOCHS + 1):
        start = time.time()

        gen_loss_avg = 0
        disc_loss_avg = 0
        batch_counter = 0

        # Execute training step for each image batch
        for image_batch in training_instance.train_dataset:
            gen_loss, disc_loss = training_instance.emg.train_step(image_batch)

            gen_loss_avg += gen_loss
            disc_loss_avg += disc_loss
            batch_counter += 1

        gen_loss_avg = gen_loss_avg / batch_counter
        disc_loss_avg = disc_loss_avg / batch_counter

        # Save images for the example SEEDs we have set (@ output/images/)
        training_instance.emg.generate_and_save_images(epoch + 1)

        # Save the model every 50 epochs
        if epoch % 50 == 0:
            training_instance.checkpoint.save(file_prefix=training_instance.checkpoint_prefix)

        print(f'Epoch {format(epoch + 1, "4")}, Time {format(time.time() - start, ".2f")} sec, ' +
              f'Gen loss: {format(gen_loss_avg, ".4f")}, '
              f'Disc loss: {format(disc_loss_avg, ".4f")}')

        # GUI: Update image canvas in training tab (tab 2) after each episode of training
        if use_gui:
            gui_instance.training_image_canvas_update(path='output/images/image_at_epoch_{:04d}.png'.format(epoch),
                                                      progress_text_update=f"{epoch}/{training_instance.EPOCHS}")

        # GUI: Break training loop if training button is pressed again
        if use_gui and training_instance.training_flag is False:
            break

    # Save images for the example SEEDs we have set for final epoch (@ output/images/)
    training_instance.emg.generate_and_save_images(training_instance.EPOCHS)

    print(f'\nTRAINING FINISHED (Time: {format(time.time() - training_instance.train_time, ".2f")} sec)')

    # --------------- CREATE GIF --------------- #
    gif.create_gif(f'output/images/image*.png', f'output/emojigan.gif')

    # GUI: Exit while loop if we do not use GUI
    if not use_gui:
        break
