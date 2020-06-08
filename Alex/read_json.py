import json

def read_json():

    with open(f'../emoji-data/emoji.json') as json_file:
        data = json.load(json_file)
        return data
