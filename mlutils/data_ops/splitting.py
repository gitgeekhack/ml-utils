import math

import numpy as np

from mlutils.business_rule_exceptions import InvalidSplittingValues, InsufficientData
from mlutils.data_ops.dataset import Dataset

__all__ = ['split_data']


def split_data(data, train=.70, valid=.20, unseen_test=0):
    """
    This method splits the given data into training, validation and unseen testing datasets based on
    the splitting ratio provided as an input for each dataset.
       Parameter:
           data: List or numpy array
           train: number of percent data that needs to be consider for training
           valid: number of percent data that needs to be consider for valid
           unseen_test: number of percent data that needs to be consider for unseen_test

           train, valid, unseen_test values must be greater than 0 and less than 1.
           sum of teh train, valid, unseen_test values must be greater 1.
       Return:
           Dataset object with the train, valid, unseen_test dataset based on the splitting ratio.
       """
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
