from tkinter import *
from tkinter import ttk
import random
import os
import threading


class Gui(threading.Thread):

    def __init__(self, training_instance):
        threading.Thread.__init__(self)
        self.root = None
        self.texts = None
        self.texts_defaults = None
        self.texts_labels = None
        self.entries = None
        self.image_canvas = None
        self.image_on_canvas = None
        self.training_instance = training_instance
        self.progress_text = None
        self.start()

    def callback(self):
        self.root.quit()

    def button_func(self):
        self.training_instance.NOISE_DIM = int(self.entries[2].get())
        self.training_instance.EPOCHS = int(self.entries[4].get())
        self.training_instance.BATCH_SIZE = int(self.entries[3].get())
        self.training_instance.GEN_LR = float(self.entries[0].get())
        self.training_instance.DISC_LR = float(self.entries[1].get())
        self.training_instance.RESTORE_CHECKPOINT = self.entries[5].get()
        self.training_instance.initialize()

    def image_canvas_update(self, path, progress_text_update):
        if not os.path.exists(path):
            print("Image canvas path not existent!")
        else:
            img = PhotoImage(file=path)
            self.image_canvas.itemconfig(self.image_on_canvas, image=img)
            self.progress_text.delete(1.0, END)
            self.progress_text.insert(END, progress_text_update)
            self.root.update()

    def build_gui(self):
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
                      'Iterations', 'Restore Checkpoint']
        self.texts_defaults = ['2e-4', '2e-5', '100', '64', '1000', 'False']
        self.texts_labels = range(len(self.texts))
        self.entries = [Entry(tab1) for t in self.texts]

        for label in self.texts_labels:
            # Colors
            ct = [random.randrange(256) for x in range(3)]
            ct_hex = "%02x%02x%02x" % tuple(ct)
            bg_colour = '#' + "".join(ct_hex)

            # Labels for training hyperparameters
            lab = Label(tab1, text=self.texts[label], bg=bg_colour)
            lab.place(x=20, y=30 + label*60, width=200, height=50)
            self.entries[label].place(x=250, y=30 + label*60, width=200, height=50)
            self.entries[label].insert(0, self.texts_defaults[label])

        # Canvas to display progress every now and then
        self.image_canvas = Canvas(tab2, width=500, height=500)
        self.image_canvas.pack()
        self.image_canvas.place(x=50-self.image_canvas.winfo_width()/2,
                                y=150-self.image_canvas.winfo_height()/2)
        self.image_on_canvas = self.image_canvas.create_image(250, 250, image=None)

        # Training button
        Button(tab2, text='Run training', command=self.button_func, width=20).place(x=200, y=50)
        progress_label = Label(tab2, text='Progress:').place(x=200, y=100)
        self.progress_text = Text(tab2, height=1, width=15)
        self.progress_text.place(x=275, y=100)

        # Update gui
        self.root.update()

