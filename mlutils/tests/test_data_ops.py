import numpy as np

from mlutils.business_rule_exceptions import InvalidSplittingValues, InsufficientData
from mlutils.data_ops import split_data


class TestDataOpsSplitDataset():
    def test_split_data_valid_list1(self):
        dataset = split_data(list(range(10)), train=0.7, valid=0.3)
        assert len(dataset.train) == 7
        assert len(dataset.valid) == 3

    def test_split_data_valid_list2(self):
        dataset = split_data(list(range(10)), train=0.7, valid=0.2, unseen_test=0.1)
        assert len(dataset.train) == 7
        assert len(dataset.valid) == 2
        assert len(dataset.unseen_test) == 1

    def test_split_data_valid_2d_list(self):
        dataset = split_data(list([y for y in list(list(range(10)) for x in range(10))]), train=0.7, valid=0.2,
                             unseen_test=0.1)
        assert len(dataset.train) == 7
        assert len(dataset.valid) == 2
        assert len(dataset.unseen_test) == 1

    def test_split_data_valid_3d_list(self):
        dataset = split_data(list([z for z in list(list(range(10)) for y in list(list(range(10)) for x in range(10)))]),
                             train=0.7, valid=0.2, unseen_test=0.1)
        assert len(dataset.train) == 7
        assert len(dataset.valid) == 2
        assert len(dataset.unseen_test) == 1

    def test_split_data_valid_1d_np_array(self):
        dataset = split_data(np.zeros((10,)), train=0.7, valid=0.2, unseen_test=0.1)
        assert len(dataset.train) == 7
        assert len(dataset.valid) == 2
        assert len(dataset.unseen_test) == 1

    def test_split_data_valid_2d_np_array(self):
        dataset = split_data(np.zeros((10, 5)), train=0.7, valid=0.2, unseen_test=0.1)
        assert len(dataset.train) == 7
        assert len(dataset.valid) == 2
        assert len(dataset.unseen_test) == 1

    def test_split_data_valid_3d_np_array(self):
        dataset = split_data(np.zeros((10, 5, 10)), train=0.7, valid=0.2, unseen_test=0.1)
        assert len(dataset.train) == 7
        assert len(dataset.valid) == 2
        assert len(dataset.unseen_test) == 1

    def test_split_data_valid_nd_np_array(self):
        dataset = split_data(np.zeros((10, 5, 10, 2)), train=0.7, valid=0.2, unseen_test=0.1)
        assert len(dataset.train) == 7
        assert len(dataset.valid) == 2
        assert len(dataset.unseen_test) == 1

    def test_split_data_invalid_ratio1(self):
        try:
            split_data(list(range(10)), train=0.7, valid=0.4)
        except InvalidSplittingValues as e:
            assert True

    def test_split_data_invalid_ratio2(self):
        try:
            split_data(list(range(10)), train=0.7, valid=0.3, unseen_test=0.1)
        except InvalidSplittingValues as e:
            assert True

    def test_split_data_invalid_empty_list(self):
        try:
            split_data([], train=0.7, valid=0.2, unseen_test=0.1)
        except InsufficientData as e:
            assert True

    def test_split_data_invalid_insufficient_data(self):
        try:
            split_data(list(range(3)), train=0.7, valid=0.3)
        except InsufficientData as e:
            assert True
