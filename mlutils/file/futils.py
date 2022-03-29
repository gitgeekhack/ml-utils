import os
import shutil


def get_files_from_dir(source_path):
    dir_list = os.listdir(source_path)
    return dir_list


def make_dir(target_path):
    if not os.path.exists(target_path):
        os.mkdir(target_path)


def copy_file(source_path, target_path, files=None):
    """
    This is method copies files from source_path to target_path,
    if files are not mentioned then all files from source is copied to target
    Args:
        source_path: <string> path for source directory.
        target_path: <string> path of target directory.
        files: <list> list of files to be copied from source to target.
    """

    make_dir(target_path)
    if files is None:
        files = get_files_from_dir(source_path)
    for file in files:
        shutil.copy(os.path.join(source_path, file), target_path)
