import os


def get_is_downloaded(folder_path: str, folder_name: str):
    path = os.path.join(folder_path, folder_name)

    # 1. Check if the directory exists, if not, return false
    if not os.path.exists(path):
        return False

    # 2. If directory exists, check if it isn't empty
    if os.listdir(path):
        return True
    else:
        return False
