# EmojiGenerator
EmojiGenerator for the Deep Generative Model Class @ TU Darmstadt 2020

### Prerequisits
- Install the git repository `https://github.com/iamcal/emoji-data` at toplevel into the folder `emoji-data`
- Set your working directory path to `EmojiGenerator` (toplevel) 

### TODO bis 15.6.

Allgemein:
- Literatur review why only one emoji is learned => Daniel
- Parameter search try and error => Daniel

Alex:
- Generalizing the learning class (Adam, loss, etc.)
- Single files for preprocessing and tf.dataset generation
- Try to recreate learning network

Do we have enough emojis?
- data augmentation
- can we find more emojis somewhere

Testing:
- Is it significant slower to train on bigger images
- Wasserstein => Yannik

Extensions:
- emoji2vec
- style transfer (pix2pix) => Tim bis nächsten Montag
- drawing emoji before generation
- labeling emojis (eyes, mouth, ...)
