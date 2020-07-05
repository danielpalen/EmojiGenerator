from tkinter import *
import random
import os
import threading


class GUI(threading.Thread):

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
        self.root.geometry("1280x720+30+30")

        self.texts = ['Learning Rate Generator', 'Learning Rate Discriminator', 'Noise Dimension', 'Batch Size',
                      'Iterations', 'Restore Checkpoint']
        self.texts_defaults = ['2e-4', '2e-5', '100', '64', '1000', 'False']
        self.texts_labels = range(len(self.texts))
        self.entries = [Entry(self.root) for t in self.texts]

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
        self.image_canvas = Canvas(self.root, width=500, height=500)
        self.image_canvas.pack()
        self.image_canvas.place(x=550-self.image_canvas.winfo_width()/2,
                                y=150-self.image_canvas.winfo_height()/2)
        # img = PhotoImage(file='output/images/image_at_epoch_0001.png')
        self.image_on_canvas = self.image_canvas.create_image(250, 250, image=None)

        Button(self.root, text='Run training', command=self.button_func, width=20).place(x=700, y=50)
        progress_label = Label(self.root, text='Progress:').place(x=700, y=100)
        self.progress_text = Text(self.root, height=1, width=15)
        self.progress_text.place(x=775, y=100)

        # Run gui
        # self.root.mainloop()
        self.root.update()

