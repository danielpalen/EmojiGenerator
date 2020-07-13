# EmojiGenerator
EmojiGenerator for the Deep Generative Model Class @ TU Darmstadt 2020

### Prerequisits
- Install the git repository `https://github.com/iamcal/emoji-data` at toplevel into the folder `emoji-data`
- Set your working directory path to `EmojiGenerator` (toplevel) 

### Networks
- Our project contains two networks: DCGAN and Pix2Pix
- DCGAN is a Generative Adversarial Net and learns to produce emojis from noise
- Pix2Pix learns to turn black & white emojis into color

### Training DCGAN
- For training the DCGAN net, we use a GUI (path = dcgan/main.py)
- You can set most of the hyperparameters in the GUI
- You can also start/stop/continue training while using the GUI
- The GUI gives a visual output for some fixed noise vectors, so that one can follow the training process

### Training Pix2Pix
- For training pix2pix, one has to execute `pix2pix/main.py`

### GUI: conversion DCGAN to Pix2Pix
- In the GUI it is possible to sample a picture from the trained DCGAN net
- After sampling, we can convert it with DCGAN
- For this, the model of the Pix2Pix network is needed
- Either train Pix2Pix and the model will be saved automatically
- Or download the model (contact us for a link) and put it at `output/pix2pix_model.h5`