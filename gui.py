from tkinter import *
import random

# Root node
root = Tk()
root.geometry("1000x750+30+30")

texts = ['Learning Rate Generator', 'Learning Rate Discriminator', 'Iterations']
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

# Button to run algo
Button(root, text='Run training', command=..., width=20).place(x=675, y=50)
progress_label = Label(root, text='Progress:').place(x=675, y=100)
progess_text = Text(root, height=1, width=15).place(x=750, y=100)

# Canvas to display progress every now and then
pic = Canvas(root, width=500, height=500)
pic.pack()
pic.place(x=500-pic.winfo_width()/2-250, y=375-pic.winfo_height()/2-150)
img = PhotoImage(file='output/images/image_at_epoch_0001.png')
pic.create_image(250, 250, image=img)

# Run gui
root.mainloop()
