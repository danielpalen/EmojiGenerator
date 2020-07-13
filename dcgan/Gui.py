from tkinter import *
from tkinter import ttk
import os
import threading
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from utilities.helper import predict_image_pix2pix


class Gui(threading.Thread):

    def __init__(self, training_instance):
        threading.Thread.__init__(self)
        self.root = None
        self.texts = None
        self.texts_defaults = None
        self.texts_labels = None
        self.entries = None
        self.training_image_canvas = None
        self.image_on_training_canvas = None
        self.sample_image_canvas = None
        self.image_on_sample_canvas = None
        self.update_sample_canvas_flag = False
        self.training_instance = training_instance
        self.pix2pix_instance = None
        self.progress_text = None
        self.start()

    def callback(self):
        self.root.quit()

    def tab1_button_func(self):
        """
            Defines what happens when the button in tab 1 is pressed.
        """
        self.training_instance.NOISE_DIM = int(self.entries[2].get())
        self.training_instance.EPOCHS = int(self.entries[4].get())
        self.training_instance.BATCH_SIZE = int(self.entries[3].get())
        self.training_instance.GEN_LR = float(self.entries[0].get())
        self.training_instance.DISC_LR = float(self.entries[1].get())
        self.training_instance.PIXEL_SIZE = int(self.entries[5].get())
        self.training_instance.EXAMPLE_SIZE = int(self.entries[6].get())
        self.training_instance.COLORSPACE = self.entries[7].get()
        self.training_instance.RESTORE_CHECKPOINT = self.entries[8].get()
        self.training_instance.initialize()

    def tab2_button_func(self):
        """
            Defines what happens after pressing the button in tab 2.
            Sets training flag to true / false to start or stop the training loop.
        """
        self.training_instance.training_flag = not self.training_instance.training_flag

    def tab3_button_dcgan(self):
        """
            Defines what happens when pressing the above button in tab 3.
            Samples from DCGAN, then saves sample to .png
            Then displays .png in greyscale.
        """
        filepath = f'output/generator_sample.png'
        self.training_instance.load_ckpt_and_sample_img(filepath)
        print(f'SAMPLE IMAGE CREATED WITH DCGAN!')
        img = mpimg.imread(filepath)
        plt.figure()
        plt.imshow(img)
        plt.show()

    def tab3_button_pix2pix(self):
        """
            Defines what happens when pressing the below button in tab 3.

            Reads current image from tab3 image canvas and puts it in the pix2pix generator.
            Then displays the result in the image canvas.
        """
        if self.training_instance.emg is None or self.training_instance.emg.generator_sample is None:
            print(f'Please sample DCGAN before applying pix2pix !')
        else:
            print("REQUESTED PIX2PIX.")
            img = np.asarray(self.training_instance.emg.generator_sample)

            # Convert color into gray
            if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
                NotImplementedError()
            elif len(img.shape) == 3 and img.shape[2] == 3:  # RGB
                img = np.mean(img, axis=2)
                print(img.shape)

            # Extend grayscale image from 1 to 3 dimensions
            img = np.expand_dims(img, axis=2)
            _img = img.copy()
            img = np.concatenate((img, _img, _img), axis=2)
            assert img.shape[2] == 3

            print("PIX2PIX PREDICTION ... ", end=f'')
            pred = predict_image_pix2pix(image=img, model_path='output/pix2pix_model.h5').numpy()
            print("DONE!")
            plt.figure()
            plt.imshow(pred)
            plt.axis('off')

            plt.savefig(f'output/dcgan_to_emoji2emoji_sample.png')
            plt.show()
            # plt.close()
            print(f'Sample image converted into color with pix2pix!')

    def training_image_canvas_update(self, path, progress_text_update):
        """
            Updates the image canvas in tab 1.
        """
        if not os.path.exists(path):
            pass
        else:
            img = PhotoImage(file=path)
            self.training_image_canvas.itemconfig(self.image_on_training_canvas, image=img)
            self.progress_text.delete(1.0, END)
            self.progress_text.insert(END, progress_text_update)
            self.root.update()

    def build_gui(self):
        """
            Initializes and builds all elements of the gui.
            Then updates the root node once to display it.
        """
        # Root node
        self.root = Tk()
        self.root.geometry("640x720+30+30")
        self.root.title("Emoji-GAN")
        self.root.resizable(width=False, height=False)

        # Tabs
        tab_parent = ttk.Notebook(self.root)
        tab1 = ttk.Frame(tab_parent)
        tab2 = ttk.Frame(tab_parent)
        tab3 = ttk.Frame(tab_parent)
        tab_parent.add(tab1, text='Hyperparameters')
        tab_parent.add(tab2, text='Training')
        tab_parent.add(tab3, text='Pix2Pix')
        tab_parent.pack(expand=1, fill='both')

        # Hyperparameter entry widgets
        self.texts = ['Learning Rate Generator', 'Learning Rate Discriminator', 'Noise Dimension', 'Batch Size',
                      'Iterations', 'Pixel Size', 'Example Size', 'Colorspace', 'Restore Checkpoint']
        self.texts_defaults = ['2e-4', '2e-5', '100', '64', '1000', '32', '16', f'RGB', 'True']
        self.texts_labels = range(len(self.texts))
        self.entries = [Entry(tab1) for t in self.texts]

        for label in self.texts_labels:
            # Labels for training hyperparameters
            lab = Label(tab1, text=self.texts[label], bg=f'#84ACBA')
            lab.place(x=20, y=30 + label*60, width=200, height=50)
            self.entries[label].place(x=250, y=30 + label*60, width=200, height=50)
            self.entries[label].insert(0, self.texts_defaults[label])

        # Canvas to display progress in the training tab
        self.training_image_canvas = Canvas(tab2, width=500, height=500)
        self.training_image_canvas.pack()
        self.training_image_canvas.place(x=50 - self.training_image_canvas.winfo_width() / 2,
                                         y=150 - self.training_image_canvas.winfo_height() / 2)
        self.image_on_training_canvas = self.training_image_canvas.create_image(250, 250, image=None)

        # Canvas to display progress in the tab 3
        self.sample_image_canvas = Canvas(tab3, width=500, height=500)
        self.sample_image_canvas.pack()
        self.sample_image_canvas.place(x=50 - self.training_image_canvas.winfo_width() / 2,
                                       y=150 - self.training_image_canvas.winfo_height() / 2)
        self.image_on_sample_canvas = self.sample_image_canvas.create_image(250, 250, image=None)

        # Initialization button
        Button(tab1, text='initialize', command=self.tab1_button_func,
               width=40).place(x=150, y=30 + (len(self.texts_labels) + 1)*60)

        # Training button
        Button(tab2, text='Run training', command=self.tab2_button_func, width=20).place(x=200, y=50)
        progress_label = Label(tab2, text='Progress:').place(x=200, y=100)
        self.progress_text = Text(tab2, height=1, width=15)
        self.progress_text.place(x=275, y=100)

        # DCGAN sample button
        Button(tab3, text='Sample DCGAN', command=self.tab3_button_dcgan, width=40).place(x=150, y=50)

        # Pix2Pix sample button
        Button(tab3, text='Convert by Pix2Pix', command=self.tab3_button_pix2pix, width=40).place(x=150, y=100)

        # Update gui
        self.root.update()

