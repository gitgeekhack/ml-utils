import os
import shutil
from glob import glob
from os.path import join

from tqdm import tqdm

from mlutils.exceptions import DirectoryNotFound, FileNotFound


def get_files_from_dir(dir):
    dir_list = os.listdir(dir)
    return dir_list


def get_all_absolute_file_paths_from_dir(dir):
    return [x for x in glob(join(dir, "**/*"), recursive=True) if not os.path.isdir(x)]


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def directory_exists(path):
    if not os.path.exists(path):
        raise DirectoryNotFound(f'Unable to find required directory', dir_path=path)


def file_exists(path):
    if not os.path.exists(path):
        raise FileNotFound(f'Unable to find required file', file_path=path)


def copy_file(source_path, target_path, files=None):
    """
    This is method copies files from source_path to target_path,
    if files are not mentioned then all files from source is copied to target
    Args:
        source_path: <string> path for source dir.
        target_path: <string> path of target dir.
        files: <list> list of files to be copied from source to target.
    """

    make_dir(target_path)
    if files is None:
        files = get_files_from_dir(source_path)
    for file in tqdm(files, desc=f"Copying files to {target_path.split('/')[-1]}"):
        shutil.copy(os.path.join(source_path, file), target_path)