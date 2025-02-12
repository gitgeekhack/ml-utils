from mlutils.exceptions import DirectoryNotFound, FileNotFound
from mlutils.image.yolov5 import read_label_classes, split_dataset_by_labels, dataset_summary

target_20 = './data/target_20'
source_20 = './data/source_20'


class TestYoloV5:
    def test_read_class_labels_valid(self):
        x = read_label_classes(label_path='./data/class_test.txt')
        assert x == ['class1', 'class2', 'class3', 'class4']

    def test_read_class_labels_invalid(self):
        try:
            x = read_label_classes(label_path='/data/class_test.txt')
        except FileNotFoundError as e:
            assert True

    def test_split_dataset_by_labels_invalid(self):
        try:
            x = split_dataset_by_labels(image_path='', annotation_path='', class_labels=['a'], target_path=None,
                                        save=False)
        except FileNotFoundError as e:
            assert True

    def test_summary_invalid(self):
        try:
            x = dataset_summary(data_file='', save=False)
        except FileNotFoundError as e:
            assert True

    def test_split_dataset_by_labels_valid(self):
        x = split_dataset_by_labels(image_path='data/yolov5_dataset/train/images',
                                    annotation_path='data/yolov5_dataset/train/labels/',
                                    class_labels=['class1', 'class2', 'class3', 'class4'], target_path=None, save=False)
        y = {'class1': {'3.jpeg'}, 'class2': {'3.jpeg', '1.jpeg'}, 'class3': {'2.jpeg', '1.jpeg'},
             'class4': {'3.jpeg', '2.jpeg'}}
        assert x == y
