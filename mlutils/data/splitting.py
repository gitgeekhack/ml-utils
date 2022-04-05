import math
import os
import random as rand
import numpy as np
from tqdm import tqdm
from mlutils.exceptions import InvalidSplittingValues, InsufficientData, DirectoryNotFound
from mlutils.data.dataset import Dataset
from mlutils.file.utils import copy_file, get_files_from_dir, make_dir

__all__ = ['split_data']


def split_data(data, train=.70, valid=.20, unseen_test=0.0, random=True):
    """
    This method splits the given data into training, validation and unseen testing datasets based on
    the splitting ratio provided as an input for each dataset.
       Args:
           data: List or numpy array
           train: number of percent data that needs to be considered for training
           valid: number of percent data that needs to be considered for valid
           unseen_test: number of percent data that needs to be considered for unseen_test
           random: if True it will randomly shuffle data

           train, valid, unseen_test values must be greater than 0 and less than 1.
           sum of teh train, valid, unseen_test values must be greater 1.
       Return:
           Dataset object with the train, valid, unseen_test dataset based on the splitting ratio.
       """
    if random is True: rand.shuffle(data)
    if train >= 1 or valid >= 1 or unseen_test >= 1 or not math.isclose(sum([train, valid, unseen_test]), 1):
        raise InvalidSplittingValues({'train': train, 'valid': valid, 'unseen_test': unseen_test})
    if unseen_test != 0:
        unseen_ratio = int(unseen_test * len(data))
        valid_ratio = int((valid + unseen_test) * len(data))
        if unseen_ratio == 0 or valid_ratio == 0:
            raise InsufficientData(f'Unable to Split data with length of {len(data)}')
        unseen_test, valid, train = np.split(data, [unseen_ratio,
                                                    valid_ratio])
        return Dataset({'train': train,
                        'valid': valid,
                        'unseen_test': unseen_test})
    else:
        valid_ratio = int(valid * len(data))
        if valid_ratio == 0:
            raise InsufficientData(f'Unable to Split data with length of {len(data)}')
        valid, train = np.split(data, [valid_ratio])

        return Dataset({'train': train,
                        'valid': valid})


def split_dataset_from_dir(source_path, target_path, train=0.7, unseen_test=0.3, valid=0.0, random=True):
    """
    This method takes source directory path and splits the given data into training, validation and unseen testing
    datasets based on the splitting ratio provided as an input for each dataset and saves in the target directory.
       Args:
           source_path: path for source directory.
           target_path: path of target directory
           train: number of percent data that needs to be considered for training
           valid: number of percent data that needs to be considered for valid
           unseen_test: number of percent data that needs to be considered for unseen_test

           train, valid, unseen_test values must be greater than 0 and less than 1.
           sum of teh train, valid, unseen_test values must be greater 1.
       """
    make_dir(target_path)
    if os.path.exists(source_path):
        files = get_files_from_dir(source_path)
        data = split_data(files, train=train, valid=valid, unseen_test=unseen_test, random=random)
        copy_file(source_path, os.path.join(target_path, 'train'), files=data.train)
        copy_file(source_path, os.path.join(target_path, 'valid'), files=data.valid)
        copy_file(source_path, os.path.join(target_path, 'test'), files=data.unseen_test)
    else:
        raise DirectoryNotFound(f'Unable to find source directory')
