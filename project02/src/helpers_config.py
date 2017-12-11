import json


def load_config(path="config.json"):
    return json.load(open(path))


def save_config(path, config):
    assert type(config) is dict
    with open(path, 'w') as outfile:
        json.dump(config, outfile, indent=2, sort_keys=True)

