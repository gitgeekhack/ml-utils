import os
import shutil

from mlutils.exceptions import DirectoryNotFound
from mlutils.image.yolov5 import get_bbox_by_label, read_class_labels, split_dataset_by_labels, summary

target_20 = './data/target_20'
source_20 = './data/source_20'


class TestYoloV5:
    def test_read_class_labels_valid(self):
        x = read_class_labels(label_path='./data/class_test.txt')
        assert x == ['class1', 'class2', 'class3', 'class4']

    def test_read_class_labels_invalid(self):
        try:
            x = read_class_labels(label_path='/data/class_test.txt')
        except DirectoryNotFound as e:
            assert True

    def test_split_dataset_by_labels_invalid(self):
        try:
            x = split_dataset_by_labels(image_path='', annotation_path='', class_labels=['a'],
                                        target_path=None, save=False)
        except DirectoryNotFound as e:
            assert True

    def test_summary_invalid(self):
        try:
            x = summary(data_file='', save=False)
        except DirectoryNotFound as e:
            assert True
