from tkinter import *
from Yannik.gantest import *
import random
import os


class GUI:
    def __init__(self):
        # GAN Training object
        self. new_training = EmojiGANTraining()

        # Root node
        self.root = Tk()
        self.root.geometry("1280x720+30+30")

        self.texts = ['Learning Rate Generator', 'Learning Rate Discriminator', 'Noise Dimension', 'Batch Size',
                      'Iterations']
        self.texts_defaults = ['2e-4', '2e-5', '100', '64', '1000']
        self.texts_labels = range(len(self.texts))
        self.entries = [Entry(self.root) for t in self.texts]

        self.image_canvas = None
        self.image_on_canvas = None

        self.build_gui()

    def build_gui(self):
        for label in self.texts_labels:
            # Colors
            ct = [random.randrange(256) for x in range(3)]
            ct_hex = "%02x%02x%02x" % tuple(ct)
            bg_colour = '#' + "".join(ct_hex)

            # Labels for training hyperparameters
            lab = Label(self.root, text=self.texts[label], bg=bg_colour)
            lab.place(x=20, y=30 + label*60, width=200, height=50)
            self.entries[label].place(x=250, y=30 + label*60, width=200, height=50)
            self.entries[label].insert(0, self.texts_defaults[label])

        # Canvas to display progress every now and then
        image_canvas = Canvas(self.root, width=500, height=500)
        image_canvas.pack()
        image_canvas.place(x=500-image_canvas.winfo_width()/2, y=375-image_canvas.winfo_height()/2-150)
        img = PhotoImage(file='output/images/image_at_epoch_0001.png')
        image_on_canvas = image_canvas.create_image(250, 250, image=img)

        Button(self.root, text='Run training', command=self.button_func, width=20).place(x=675, y=50)
        progress_label = Label(self.root, text='Progress:').place(x=675, y=100)
        progess_text = Text(self.root, height=1, width=15).place(x=750, y=100)

        # Run gui
        self.root.mainloop()

    # Button to run algo
    def button_func(self):
        self.new_training.GEN_LR = float(self.entries[0].get())
        self.new_training.DISC_LR = float(self.entries[1].get())
        self.new_training.NOISE_DIM = int(self.entries[2].get())
        self.new_training.BATCH_SIZE = int(self.entries[3].get())
        self.new_training.EPOCHS = int(self.entries[4].get())
        self.new_training.training()


if __name__ == "__main__":
    new_gui = GUI()
    new_gui.build_gui()

