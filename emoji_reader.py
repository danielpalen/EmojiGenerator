import os
import json
import matplotlib.pyplot as plt
import imageio
import glob

import numpy as np
import tensorflow as tf


class EmojiReader:

    def __init__(self):
        self.emojis_base_path = os.path.join('.', 'emoji-data')
        with open(os.path.join(self.emojis_base_path, 'emoji.json')) as f:
            self.emoji_meta_data = json.load(f)

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

    def read_images_from_sheet(self):
        raise NotImplementedError

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


if __name__ == '__main__':
    reader = EmojiReader()
    images = reader.read_images()
    print()
