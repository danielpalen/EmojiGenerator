import os
import json
import matplotlib.pyplot as plt
import imageio
import glob

import numpy as np
import tensorflow as tf


class EmojiReader:
    """
    This class reads emojis from the https://github.com/iamcal/emoji-data
    emoji data collection. For this to work, please download the data
    into a folder 'emoji-data' that is at the same level as this file.
    """

    def __init__(self, categories):
        """
        Create an emoji reader object that reads the meta data from
        emoji.json and selects certain entries.
        :param categories: list of emoji categories we want to include.
            Possible categories: {'Travel & Places', 'Objects',
            'Animals & Nature', 'Skin Tones', 'Activities', 'Symbols',
            'People & Body', 'Food & Drink', 'Flags', 'Smileys & Emotion'}
        """
        # Load emoji meta data
        self.emojis_base_path = os.path.join('.', 'emoji-data')
        with open(os.path.join(self.emojis_base_path, 'emoji.json')) as f:
            self.emoji_meta_data = json.load(f)

        # Categories
        if not categories:
            pass # If empty, keep all categories
        else:
            self.emoji_meta_data


    def read_images(self, filter=None):  # Filter could be a filter functino to select s
        # images = [{
        #     'meta': d,
        #     'image': self._read_single_image(os.path.join(self.emojis_base_path, 'img-apple-64'), d['image'])
        # } for d in self.emoji_meta_data]

        # TODO: return as tf.data.DataSet

        images = [
            self._read_single_image(os.path.join(
                self.emojis_base_path, 'img-apple-64'), d['image'])
            for d in self.emoji_meta_data
        ]
        images = [i for i in images if i is not None]
        images = np.stack(images, axis=0).astype('float32')

        # TODO: pre processing here.

        images = tf.data.Dataset.from_tensor_slices(images)
        return images

    def _read_single_image(self, image_base_path, image_name):
        """Read emoji image and return as np array"""
        # TODO: is there a more allegant way to do this?
        try:
            image_path = os.path.join(image_base_path, image_name)
            image_data = imageio.imread(image_path)
            #print(image_name, image_data.shape)
            return image_data
        except:
            print(
                f'Error: Could not read image {os.path.join(image_base_path, image_name)}')
            return None

    def read_images_from_sheet(self):
        raise NotImplementedError


if __name__ == '__main__':
    reader = EmojiReader()
    images = reader.read_images()
    print()
