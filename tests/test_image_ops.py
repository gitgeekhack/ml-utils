import os
import shutil

from mlutils.data.filter import filter_images_by_dimension

target_20 = './data/target_20'
source_20 = './data/source_20'


class TestDataOpsSplitDataset():
    def test_filter_dataset_by_dimension_valid(self):
        filter_images_by_dimension(source_20, target_20, 150, 150)
        assert sum(len(files) for _, _, files in os.walk(target_20)) == 6
        if os.path.exists(target_20):
            shutil.rmtree(target_20)

    def test_filter_dataset_by_dimension_valid_2(self):
        filter_images_by_dimension(source_20, target_20, 0, 0)
        assert sum(len(files) for _, _, files in os.walk(target_20)) == 10
        if os.path.exists(target_20):
            shutil.rmtree(target_20)
