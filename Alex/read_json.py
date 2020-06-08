import json

def read_json():

    with open(f'emoji-datasource-apple/emoji.json') as json_file:
        data = json.load(json_file)
        return data
