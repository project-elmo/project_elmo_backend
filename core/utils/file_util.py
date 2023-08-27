import os
import shutil


def get_is_downloaded(folder_path: str, folder_name: str):
    """
    Check if the file is already downloaded.
    Return True if the folder exists and contains files,
    Return False otherwise.
    """
    path = os.path.join(folder_path, folder_name)

    # 1. Check if the directory exists, if not, return false
    if not os.path.exists(path):
        return False

    # 2. If directory exists, check if it isn't empty
    if os.listdir(path):
        return True
    else:
        return False


def copy_file(source_path, destination_dir):
    if not os.path.exists(source_path):
        print(f"Source file {source_path} does not exist")
        return False

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Build the full destination path, including the filename
    destination_path = os.path.join(destination_dir, os.path.basename(source_path))

    shutil.copy2(source_path, destination_path)

    if os.path.exists(destination_path):
        print(f"File successfully copied to {destination_path}")
        return True
    else:
        print(f"Failed to copy the file to {destination_path}")
        return False


def is_valid_file_path(path_str):
    """
    Check if the given string is a valid file path and exists in the filesystem.
    """
    print("is_valid_file_path", path_str)
    return os.path.isfile(path_str)


def get_filename_from_path(path):
    """
    Extract the filename from a given path.
    """
    return os.path.basename(path)


def get_extension_from_path(path):
    """
    Extract the file extension from a given path without the dot.
    """
    return os.path.splitext(path)[1][1:]


def is_supported_extension(filename):
    """
    Check if the filename has a supported extension.
    Returns the extension if supported, otherwise None.
    """
    _, ext = os.path.splitext(filename)

    supported_extensions = [
        ".json",
        ".csv",
        ".excel",
    ]  # Add or remove extensions as required

    if ext in supported_extensions:
        return True
    return False
