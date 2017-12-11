import json


def load_config(path="config.json"):
    json.load(open("config.txt"))


def save_config(path, config):
    assert type(config) is dict
    json.dump(config, open(path, 'w'), indent=2, sort_keys=True)

