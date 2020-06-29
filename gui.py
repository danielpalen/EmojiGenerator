from tkinter import *
from Yannik.gantest import *
import random
import os

# GAN Training object
new_training = EmojiGANTraining()

# Root node
root = Tk()
root.geometry("1280x720+30+30")

texts = ['Learning Rate Generator', 'Learning Rate Discriminator', 'Noise Dimension', 'Batch Size', 'Iterations']
defaults = ['2e-4', '2e-5', '100', '64', '1000']
labels = range(len(texts))
entries = [Entry(root) for t in texts]
for label in labels:
    # Colors
    ct = [random.randrange(256) for x in range(3)]
    ct_hex = "%02x%02x%02x" % tuple(ct)
    bg_colour = '#' + "".join(ct_hex)

    # Labels for training hyperparameters
    lab = Label(root, text=texts[label], bg=bg_colour)
    lab.place(x=20, y=30 + label*60, width=200, height=50)
    entries[label].place(x=250, y=30 + label*60, width=200, height=50)
    entries[label].insert(0, defaults[label])

# Canvas to display progress every now and then
image_canvas = Canvas(root, width=500, height=500)
image_canvas.pack()
image_canvas.place(x=500-image_canvas.winfo_width()/2, y=375-image_canvas.winfo_height()/2-150)
img = PhotoImage(file='output/images/image_at_epoch_0001.png')
image_canvas.create_image(250, 250, image=img)


# Button to run algo
def button_func():
    new_training.GEN_LR = float(entries[0].get())
    new_training.DISC_LR = float(entries[1].get())
    new_training.NOISE_DIM = int(entries[2].get())
    new_training.BATCH_SIZE = int(entries[3].get())
    new_training.EPOCHS = int(entries[4].get())
    new_training.training(image_canvas)


Button(root, text='Run training', command=button_func, width=20).place(x=675, y=50)
progress_label = Label(root, text='Progress:').place(x=675, y=100)
progess_text = Text(root, height=1, width=15).place(x=750, y=100)



# Run gui
root.mainloop()
