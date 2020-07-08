import os
from EmojiReader import EmojiReader
from utilities import constants
from utilities.emojiPrePro import *


def generate_training_images(filepath=f'output/training_images', number_images=1000, size=32, mode="grey"):
    """
    Creates training images for DCGAN and Pix2Pix. Returns them like the function
    'read_images_from_sheet' and saves them in a folder if there is
        already enough data in the folder no new data is generated and
        the content is returned.

    :param filepath: Path to folder were the training data should be saved.
    :param number_images: Number of training images the method returns.
    :param size: The pixel size of the returned images (32 or 64)
    :param mode: (grey, normal, pix2pix) For greyscale images, normal images
        or images for pix2pix
    """

    assert size in [32, 64]
    assert mode in ["grey", "pix2pix", "normal"]
    assert (not (size != 64 and mode == "pix2pix"))

    PATH = "./output/data_augmentation/"

    if not os.path.isdir(PATH):
        os.makedirs(PATH)

    folders = [f'h1', f'h2', f'h3', f'h4', f'l1', f'l2', f'l3', f'l4']
    for f in folders:
        if not os.path.isdir(PATH + f):
            os.makedirs(PATH + f)

    if mode == "pix2pix" and (not os.path.isdir(PATH + "pix2pix")):
        os.makedirs(PATH + "pix2pix")

    if not os.path.isdir(filepath):
        os.makedirs(filepath)

    # Cutting out a quarter emoji and saving it in different folders
    # Either of the high or low part of the emoji
    if len([name for name in os.listdir(PATH + 'h1') if os.path.isfile(name)]) < 10:

        reader_high = EmojiReader(databases=[f'google'], emoji_names=constants.FACE_EMOJIS_HIGH)
        images_high = reader_high.read_images_from_sheet(pixel=64, debugging=False, png_format='RGB')

        for i, image in enumerate(images_high):
            quartering(image, PATH, 'h' + str(i) + ".png", True)

        reader_low = EmojiReader(databases=[f'google'], emoji_names=constants.FACE_EMOJIS_LOW)
        images_low = reader_low.read_images_from_sheet(pixel=64, debugging=False, png_format='RGB')

        for i, image in enumerate(images_low):
            quartering(image, PATH, 'l' + str(i) + ".png", False)

    images = []

    # Create new emojis and save in "final"
    if (len([name for name in os.listdir(filepath) if
             os.path.isfile(os.path.join(filepath, name))]) < number_images):

        for x in range(number_images // 2):
            print(str(x))

            # Create pictures cut from top (high) and bottom (low) half of the image
            for pos in [f'h', f'l']:
                rand_images = []
                for i in range(1, 5):
                    _file = random.choice(os.listdir(PATH + pos + str(i) + os.sep))
                    _filename = os.fsdecode(_file)
                    rand_images.append(imageio.imread(PATH + pos + str(i) + os.sep + _filename))

                image_concat = np.concatenate(
                    (np.concatenate((rand_images[0], rand_images[1]), axis=1),
                     np.concatenate((rand_images[2], rand_images[3]), axis=1)),
                    axis=0)

                image_rotate = Image.fromarray(image_concat)
                image_rotate.rotate(random.randint(-5, 5), fillcolor='white')
                image_final = np.array(image_rotate)

                if size == 32:
                    image_final = image_final[::2, ::2]
                if mode == "grey":
                    image_to_grey(image_final)

                images.append(image_final)

                if not mode == "pix2pix":
                    print(filepath + pos + str(x) + '.png')
                    imageio.imwrite(filepath + os.sep + pos + str(x) + '.png', image_final)
                else:
                    imageio.imwrite(PATH + "pix2pix/" + pos + str(x) + '.png', image_final)

        if mode == "pix2pix":
            # Create training data and save in "train"
            for file in os.listdir(PATH + "pix2pix/"):
                filename = os.fsdecode(file)
                if filename.endswith(".png"):
                    image = imageio.imread(PATH + 'pix2pix/' + filename)
                    preProcessing(image, filepath, filename)
        else:
            return np.asarray(images, dtype=np.float32)

    else:
        images = []
        for file in os.listdir(filepath):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg"):
                images.append(imageio.imread(filepath + filename))
        return np.asarray(images, dtype=np.float32)

generate_training_images()