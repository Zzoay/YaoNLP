
import json


def _load_json(config_path):
    with open(config_path) as f:
        dct = json.load(f)
        return dct


def load_data_config():

    return


def load_model_config():

    return


def load_trainer_config():

    return


if __name__ == "__main__":
    with open("config_example/model.json") as f:
        dct = json.load(f)
        print(dct)