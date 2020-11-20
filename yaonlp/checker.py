
import os

import config_loader

def check_path(config, mode="input"):
    if mode == "input":
        not_exist_lst = []
        for key, value in config.items():
            if "path" in key or "file" in key:
                not_exist_lst.append((key, value))
        if len(not_exist_lst) > 0:
            for key, value in not_exist_lst:
                print("'{}': '{}' not exist".format(key, value))
            assert False # TODO some error
    
    elif mode == "save":
        for key, value in config.items():
            exist = True
            if "path" in key:
                exist = os.path.exists(conf)
                if not exist:
                    print("{} not exist, auto make".format(conf))
                    os.makedirs(conf)
            elif "file" in key:
                exist = os.path.exists(conf)
                # maybe 'the path not exist', or 'path exist but file not exist'
                if not exist:
                    path, file_name = os.path.split(conf)
                    path_exist = os.path.exists(path)
                    # path not exist
                    if not path_exist:
                        print("{} not exist, auto make".format(path))
                        os.makedirs(path)


if __name__ == "__main__":
    config_file = "config_example/data.json"
    config = config_loader._load_json(config_file)
    check_path(config)
