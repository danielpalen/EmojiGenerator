import os
import sys
import json
import matplotlib.pyplot as plt
import imageio
import glob

import numpy as np
import tensorflow as tf

from PIL import Image


class EmojiReader:
    """
    This class reads emojis from the https://github.com/iamcal/emoji-data
    emoji data collection. For this to work, please download the data
    into a folder 'emoji-data' that is at the same level as this file.
    """

    def __init__(self, categories=None):
        """
        Create an emoji reader object that reads the meta data from
        emoji.json and selects certain entries.
        :param categories: list of emoji categories we want to include.
            Possible categories: {'Travel & Places', 'Objects',
            'Animals & Nature', 'Skin Tones', 'Activities', 'Symbols',
            'People & Body', 'Food & Drink', 'Flags', 'Smileys & Emotion'}
        """

        # Load emoji meta data
        self.EMOJI_BASE_PATH = os.path.join('.', 'emoji-data')
        with open(os.path.join(self.EMOJI_BASE_PATH, 'emoji.json')) as f:
            self.FULL_META_DATA = json.load(f)

        # Categories
        if not categories:
            # If categories=None, we want all categories
            self.selected_meta_data = self.FULL_META_DATA
        else:
            self.selected_meta_data = []
            for i in range(len(self.FULL_META_DATA)):
                if self.FULL_META_DATA[i][f'category'] in categories:
                    self.selected_meta_data.append(self.FULL_META_DATA[i])

        print(f'Image Reader created!')
        print(f'{len(self.selected_meta_data)} meta data entries selected.')

    def read_images(self, filter=None):  # Filter could be a filter functino to select s
        # images = [{
        #     'meta': d,
        #     'image': self._read_single_image(os.path.join(self.emojis_base_path, 'img-apple-64'), d['image'])
        # } for d in self.emoji_meta_data]

        # TODO: return as tf.data.DataSet

        images = [
            self._read_single_image(os.path.join(
                self.EMOJI_BASE_PATH, 'img-apple-64'), d['image'])
            for d in self.selected_meta_data
        ]
        images = [i for i in images if i is not None]
        images = np.stack(images, axis=0).astype('float32')

        # TODO: pre processing here.

        images = tf.data.Dataset.from_tensor_slices(images)
        return images

    def _read_single_image(self, image_base_path, image_name):
        """Read emoji image and return as np array"""
        # TODO: is there a more elegant way to do this?
        try:
            image_path = os.path.join(image_base_path, image_name)
            image_data = imageio.imread(image_path)
            #print(image_name, image_data.shape)
            return image_data
        except:
            print(
                f'Error: Could not read image {os.path.join(image_base_path, image_name)}')
            return None

    def read_images_from_sheet(self, pixel, databases, in_all_db=True, png_format=f'RGB', debugging=False):
        """
        This function reads emoji images by utilizing the sheets
        that contain all emojis in the git repository instead
        of importing the individual emoji images.

        :param pixel: size of each emoji. must be in {16, 20, 32, 64}.
        :param databases: list of the databases we want to use
            (must be in [apple, facebook, google, twitter])
        :param in_all_db: boolean. determine if emoji shall only be
            included if it is present in all db
        :param png_format: the format (RGB or RGBA) of the images
        :param debugging: boolean that determines if debugging is on.
        :return: list of numpy arrays
        """

        assert pixel in [16, 20, 32, 64]
        assert all(db in [f'apple', f'facebook', f'google', f'twitter'] for db in databases)
        assert png_format in [f'RGB', f'RGBA']

        if debugging:
            os.mkdir(f'output/selected_emoji/')

        # Collect all emoji images
        images = []

        for db in databases:
            im_path = f'emoji-data/sheet_{db}_{pixel}.png'
            im = Image.open(im_path)

            # Convert to RGB if we don't want RGBA
            if png_format == f'RGB':
                im.load()  # needed for split()
                background = Image.new('RGB', im.size, (255, 255, 255))
                background.paste(im, mask=im.split()[3])  # 3 is the alpha channel
            else:
                # Because the RGB values are random at places where
                # alpha == 0, we set it manually to white
                im[im[:, :, 3] == 0] = [255, 255, 255, 0]

            im.show() if debugging else ...  # debug: show sheet
            im = np.asarray(im)

            for i in range(len(self.selected_meta_data)):
                meta_data = self.selected_meta_data[i]

                # Check if emoji exists in this database
                if not meta_data[f'has_img_{db}']:
                    continue

                # If desired, check if present in all databases
                elif in_all_db:
                    if not all(meta_data[f'has_img_{_db}'] for _db in databases):
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
                    plt.imshow(emoji_im)
                    plt.savefig(f'output/selected_emoji/emoji_{i}.png')
                    plt.show()
                    plt.clf()

        print(f'{len(images)} images extracted (img size = {pixel}, type = {png_format}, db = {databases}).')

        return images


if __name__ == '__main__':
    reader = EmojiReader(categories=['Smileys & Emotion'])
    reader.read_images_from_sheet(pixel=32, databases=[f'apple', f'twitter', f'facebook'])
