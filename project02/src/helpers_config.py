import json


def load_config(path="default_config.json"):
    """
    Load a JSON dictionary given it's path

    :param path:  where is stored the json file
    :return: the dictionary
    """
    return json.load(open(path))


def save_config(path, config):
    """
    Save a config into a json file

    :param path: where to store a file
    :param config: a dictionary
    :return: None
    """
    assert type(config) is dict
    with open(path, 'w') as outfile:
        json.dump(config, outfile, indent=2, sort_keys=True)
