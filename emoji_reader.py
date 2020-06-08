import os
import json
import matplotlib.pyplot as plt
import imageio
import glob

class EmojiReader:

    def __init__(self):
        emoji_data_base_path = './emoji.json' 
        with open(emoji_data_base_path) as f:
            self.emoji_meta_data = json.load(f)

    def read_images(self, images_base_path, filter=None): # Filter could be a filter functino to select s
        images = [{
            'meta' : d,
            'image': read_single_image(images_base_path, d['image']) 
        } for d in emoji_meta_data]
        return images

    def read_images_from_sheet(...):
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
            print(f'Error: Could not read image {os.path.join(image_base_path, image_name)}')
            return None