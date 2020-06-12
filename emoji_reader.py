import os
import sys
import json
import matplotlib.pyplot as plt
import imageio
import glob

import numpy as np
import tensorflow as tf

from PIL import Image
import shutil


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

        # Remind user to preprocess images
        self.applied_preprocessing = False

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
        assert png_format in [f'RGB', f'RGBA']

        if debugging:
            shutil.rmtree('output/selected_emoji/')
            os.mkdir(f'output/selected_emoji/')

        # Collect all emoji images
        images = []

        for db in self.databases:
            im_path = f'emoji-data/sheet_{db}_{pixel}.png'

            # Images are  as RGBA
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
                    plt.imshow(emoji_im)
                    plt.savefig(f'output/selected_emoji/emoji_{i}.png')
                    plt.show()
                    plt.clf()

        print(f'{len(images)} images extracted from sheets')
        print(f'- img size = {pixel}, type = {png_format}, db = {self.databases}.\n')

        self.images_as_np = np.asarray(images, dtype=np.float32)

        return True

    def apply_preprocessing(self):
        """
        TODO: DO we want to add an option for gaussian preprocessing?
        Convert self.images_as_np to float, make them fall around 0
        and into [-1, 1].
        :return: true if successful
        """
        images = self.images_as_np
        images = np.asarray(images, np.float32)
        images = (images - 127.5) / 127.5
        self.images_as_np = images

        self.applied_preprocessing = True

        print(f'Preprocessing:')
        print(f'- {np.average(images, axis=(0, 1, 2))} should be around 0')
        print(f'- {np.min(images, axis=(0, 1, 2))} should be around -1')
        print(f'- {np.max(images, axis=(0, 1, 2))} should be around 1\n')

        return True

    def get_tf_dataset(self, batch_size):
        """
        Create a tensorflow dataset which is shuffled and divided
        into batches. It uses the internal self.images_as_np variable.
        Make sure each entry in images_as_np represents an image.

        :param batch_size: the desired batch size
        :return: a tensorflow Dataset
        """
        if not self.applied_preprocessing:
            print(f'\x1b[1;31;43mWarning: Data not yet preprocessed.\x1b[0m\n')

        buffer_size = 60000

        # Batch and shuffle the data
        tf_dataset = tf.data.Dataset.from_tensor_slices(self.images_as_np)
        tf_dataset = tf_dataset.shuffle(buffer_size).batch(batch_size)

        return tf_dataset


if __name__ == '__main__':
    reader = EmojiReader(databases=[f'apple', f'twitter', f'facebook'], categories=['Smileys & Emotion'])
    reader.read_images_from_sheet(pixel=32)
