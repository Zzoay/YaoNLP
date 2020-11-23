
import os


def check_dirConfig(config: dict, mode="input"):
    if mode == "input":
        not_exist_lst = []
        for key, value in config.items():
            if ("path" in key or "file" in key) and not os.path.exists(value):
                not_exist_lst.append((key, value))
        if len(not_exist_lst) > 0:
            for key, value in not_exist_lst:
                print("'{}': '{}' do not exist".format(key, value))
            assert False, "path do not exist" # TODO raise some error
    
    elif mode == "save":
        for key, value in config.items():
            exist = True
            if "path" in key:
                exist = os.path.exists(value)
                if not exist:
                    print("Warning: '{}': '{}' do not exist, auto make".format(key, value))
                    os.makedirs(value)
            elif "file" in key:
                exist = os.path.exists(value)
                # maybe 'the path not exist', or 'path exist but file not exist'
                if not exist:
                    path, file_name = os.path.split(value)
                    path_exist = os.path.exists(path)
                    # path not exist
                    if not path_exist:
                        print("Warning: '{}': '{}' do not exist, auto make".format(key, path))
                        os.makedirs(path)
                else:
                    print("Warning: '{}': '{}' exists, may will be overrided".format(key, value))
