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

    def read_images_from_sheet(self, pixel, databases, type, debugging=False):
        """
        This function does not import each image individually, but utilizes
        the sheets that contain all emojis in the git repository. It reads one
        sheet (specified by pixel size and color depth) for every database
        (apple, facebook, google or twitter) specified and extracts each emoji
        from these sheet.

        :param pixel: size of each emoji. must be in {16, 20, 32, 64}.
        :param databases: list of the databases we want to use
            (must be in [apple, facebook, google, twitter])
        :param type: the format (RGB or RGBA) of the images
        :param debugging: boolean that determines if debugging is on.
        :return: list of numpy arrays
        """

        assert pixel in [16, 20, 32, 64]
        assert all(db in [f'apple', f'facebook', f'google', f'twitter'] for db in databases)
        assert type in [f'RGB', f'RGBA']

        for db in databases:
            im_path = f'emoji-data/sheet_{db}_{pixel}.png'
            img = Image.open(im_path)

            # Convert to RGB if we don't want RGBA
            if type == f'RGB':
                im.load()  # needed for split()
                background = Image.new('RGB', im.size, (255, 255, 255))
                background.paste(im, mask=im.split()[3])  # 3 is the alpha channel
                im = np.asarray(background)

            print(np.asarray(img).shape)

            img.show() if debugging else ...  # debug: show sheet

            print(img.shape)
            sys.exit(9)
            images = []

            for i in range(len(self.selected_meta_data)):
                em = self.selected_meta_data[i]
                x = em[f'sheet_x']
                y = em[f'sheet_y']

                gaps_x = 1 + x * 2
                gaps_y = 1 + y * 2

                start_x = gaps_x + pixel * x
                start_y = gaps_y + pixel * y

                # Attention: x and y axis are changed
                im = img[start_y:start_y + pixel, start_x:start_x + pixel]

                # KEEP RGBA
                # Fill white everywhere alpha == 0
                # im[im[:, :, 3] == 0] = [255, 255, 255, 0]

                # WORK WITH RGB
                im = Image.fromarray(im)


                images.append(im)
                if debugging:
                    plt.imshow(im)
                    plt.savefig(f'output/selected_emojis/emoji_{i}.png')
                    plt.show()
                    plt.clf()


if __name__ == '__main__':
    reader = EmojiReader(categories=['Smileys & Emotion'])
    reader.read_images_from_sheet(pixel=32, databases=[f'apple'], type=f'RGB', debugging=True)
