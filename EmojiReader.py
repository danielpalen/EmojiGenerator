import os
import random
import sys
import json
import matplotlib.pyplot as plt
import imageio
import glob

import numpy as np
import tensorflow as tf

from PIL import Image
import shutil

from utilities.emojiPrePro import quartering, preProcessing, image_to_grey
from utilities import constants


class EmojiReader:
    """
    This class reads emojis from the https://github.com/iamcal/emoji-data
    emoji data collection. For this to work, please download the data
    into a folder 'emoji-data' that is at the same level as this file.

    When instantiating, the user has to select the emoji that shall be used.
    By calling a function, the images are returned as numpy array.
    """

    def __init__(self, databases, in_all_db=True, categories=None, emoji_names=None):
        """
        Create an emoji reader object that reads the meta data from
        emoji.json and selects certain entries.

        :param databases: list of the databases we want to use
            (must be in [apple, facebook, google, twitter])
        :param in_all_db: boolean. determine if emoji shall only be
            included if it is present in all db
        :param categories: list of emoji categories we want to include.
            Possible categories: {'Travel & Places', 'Objects',
            'Animals & Nature', 'Skin Tones', 'Activities', 'Symbols',
            'People & Body', 'Food & Drink', 'Flags', 'Smileys & Emotion'}.
            If 'None' all categories are included
        :param emoji_names: a list of the unicode emoji names we want to
            include. If 'None', all emoji are included.
        """

        assert all(db in [f'apple', f'facebook', f'google', f'twitter'] for db in databases)
        self.databases = databases
        self.images_as_np = None

        # Load emoji meta data
        self.EMOJI_BASE_PATH = os.path.join('.', 'emoji-data')
        with open(os.path.join(self.EMOJI_BASE_PATH, 'emoji.json')) as f:
            self.FULL_META_DATA = json.load(f)

        # Select emoji by name
        if emoji_names:
            self.selected_meta_data = []
            for i in range(len(self.FULL_META_DATA)):
                name = self.FULL_META_DATA[i][f'name']
                # Some have no name, so skip all with 'None'
                if name and name.lower() in emoji_names:
                    self.selected_meta_data.append(self.FULL_META_DATA[i])
        else:
            self.selected_meta_data = self.FULL_META_DATA

        # Select emoji by category
        if categories:
            _selected_meta_data = []
            for i in range(len(self.selected_meta_data)):
                if self.selected_meta_data[i][f'category'] in categories:
                    _selected_meta_data.append(self.selected_meta_data[i])
            self.selected_meta_data = _selected_meta_data

        # Only keep emoji that are present in all selected databases (if desired)
        trashed = 0
        if in_all_db:
            _selected_meta_data = []
            for i in range(len(self.selected_meta_data)):
                if all(self.selected_meta_data[i][f'has_img_{_db}'] for _db in databases):
                    _selected_meta_data.append(self.selected_meta_data[i])
                else:
                    trashed += 1
            self.selected_meta_data = _selected_meta_data

        print(f'Image Reader created!')
        print(f'- Databases: {databases}')
        print(f'- Categories: {categories}')
        print(f'- {trashed} entries deselected, because not in all sel databases') if in_all_db else ...
        print(f'- {len(self.selected_meta_data)} meta data entries selected.\n')

    def read_images_from_sheet(self, pixel, png_format=f'RGB', debugging=False):
        """
        This function reads emoji images by utilizing the sheets
        that contain all emojis in the git repository instead
        of importing the individual emoji images.

        :param pixel: size of each emoji. must be in {16, 20, 32, 64}.
        :param png_format: the format (RGB or RGBA) of the images
        :param debugging: boolean that determines if debugging is on.
        :return: list of numpy arrays
        """

        assert pixel in [16, 20, 32, 64]
        assert png_format in [f'RGB', f'RGBA', f'gray']

        if debugging:
            if os.path.isdir(f'output/selected_emoji/'):
                shutil.rmtree(f'output/selected_emoji/')
            os.mkdir(f'output/selected_emoji/')

        # Collects all emoji images
        images = []

        for db in self.databases:
            im_path = f'emoji-data/sheet_{db}_{pixel}.png'

            # Holds the sheet with all emojis
            # First loaded as RGBA
            im = None

            if png_format == f'RGBA':
                # Because the RGB values are random at places where
                # alpha == 0, we set it manually to white
                im = np.asarray(imageio.imread(im_path))  # use imageio library
                im[im[:, :, 3] == 0] = [255, 255, 255, 0]
                im = Image.fromarray(im)
            else:
                # Use RGB
                im = Image.open(im_path)  # use PIL library
                im.load()  # needed for split()
                background = Image.new(f'RGB', im.size, (255, 255, 255))
                background.paste(im, mask=im.split()[3])  # 3 is the alpha channel
                im = background

                if png_format == f'gray':
                    im = im.convert(f'L')

            im.show() if debugging else ...  # debug: show sheet
            im = np.asarray(im)

            for i in range(len(self.selected_meta_data)):
                meta_data = self.selected_meta_data[i]

                # Check if emoji exists in this database
                if not meta_data[f'has_img_{db}']:
                    continue

                # Get the x. emoji in x direction
                x = meta_data[f'sheet_x']
                y = meta_data[f'sheet_y']

                # Calculate the empty rows in front of this emoji
                gaps_x = 1 + x * 2
                gaps_y = 1 + y * 2

                # Calculate the top left corner of this emoji
                start_x = gaps_x + pixel * x
                start_y = gaps_y + pixel * y

                # Extract emoji
                # Attention: x and y axis are changed
                emoji_im = im[start_y:start_y + pixel, start_x:start_x + pixel]

                images.append(emoji_im)

                if debugging:
                    if png_format == f'gray':
                        plt.imshow(emoji_im, cmap=f'gray')
                    else:
                        plt.imshow(emoji_im)
                    plt.savefig(f'output/selected_emoji/emoji_{db}_{i}.png')
                    plt.show()
                    plt.close()

        print(f'{len(images)} images extracted from sheets')
        print(f'- img size = {pixel}, type = {png_format}, db = {self.databases}.\n')

        self.images_as_np = np.asarray(images, dtype=np.float32)

        # We always want our data to have 4 dimensions, even for grayscale (== last dim will be 1)
        if len(self.images_as_np.shape) == 3:
            self.images_as_np = np.expand_dims(self.images_as_np, axis=3)

        return self.images_as_np


if __name__ == '__main__':
    reader = EmojiReader(databases=[f'apple', f'twitter', f'facebook', f'google'], emoji_names=constants.FACE_EMOJIS_DCGAN_TRAINING)
    reader.read_images_from_sheet(pixel=32, png_format=f'gray', debugging=True)
