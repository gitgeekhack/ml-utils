import os
import sys
import uuid

import albumentations as A
import cv2
from tqdm import tqdm

from mlutils.exceptions import InvalidConfiguration
from mlutils.file.utils import get_all_absolute_file_paths_from_dir, make_dir


class Augmentation:
    __required_keys = ['transform', 'multiplier', 'source_path', 'target_path']

    def __init__(self, config):

        for key in self.__required_keys:
            try:
                if not config[key]:
                    raise InvalidConfiguration(f'"{key}" must have a valid value')
            except KeyError:
                raise InvalidConfiguration(f'Missing required key "{key}"')

        self.transform = config['transform']
        self.multiplier = config['multiplier']
        self.source_path = config['source_path']
        self.target_path = config['target_path']
        self.target_path_images = os.path.join(self.target_path, 'images')
        make_dir(self.target_path_images)
        try:
            self.label_path = config['label_path']
            self.target_path_labels = os.path.join(self.target_path, 'labels')
            make_dir(self.target_path_labels)
        except KeyError:
            self.label_path = None

    def __load_images(self):
        images = get_all_absolute_file_paths_from_dir(self.source_path)
        return images

    def __load_labels(self):
        return get_all_absolute_file_paths_from_dir(self.label_path) if self.label_path else []

    def __map_dataset(self, images, labels):
        dataset = {}

        for file_path in images:
            _, file_name = os.path.split(file_path)
            dataset[os.path.splitext(file_name)[0]] = {'image': file_path}
        for file_path in labels:
            _, file_name = os.path.split(file_path)
            if os.path.splitext(file_name)[0] in dataset.keys():
                dataset[os.path.splitext(file_name)[0]] |= {'label': file_path}
        return dataset

    def __read_image(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __read_label(self, path):
        with open(path) as f:
            lines = f.readlines()
            bboxes = []
            _label_classes = []
            for line in lines:
                data = line.split()
                label = int(data[0])
                _label_classes.append(label)
                obj_list = []
                for d in data[1:]:
                    d = float(d)
                    obj_list.append(d)
                bboxes.append(obj_list)
        return bboxes, _label_classes

    def __save_label(self, file_name, transformed):
        with open(os.path.join(self.target_path_labels, file_name + '.txt'), "w") as write_file:
            for i in enumerate(transformed['class_labels']):
                label = i[1]
                write_file.write(str(label) + " ")
                for j in transformed['bboxes'][i[0]]:
                    string = "{:.6f}".format(j)
                    write_file.write(string + " ")
                write_file.write("\n")

    def __save_image(self, file_name, transformed):
        transformed_image = transformed["image"]
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(self.target_path_images, file_name + '.jpg'), transformed_image)

    def augment_images(self):
        images = self.__load_images()
        labels = self.__load_labels()
        dataset = self.__map_dataset(images, labels)
        for k, v in tqdm(dataset.items(), desc="Augmenting images"):
            image = self.__read_image(v['image'])
            bboxes = _label_classes = None
            if self.label_path:
                bboxes, _label_classes = self.__read_label(v['label'])
            self.__apply_transformation(_label_classes, bboxes, image)

    def __apply_transformation(self, _label_classes, bboxes, image):
        new_name = uuid.uuid1().hex
        for i in range(self.multiplier):
            i_new_name = new_name + f'_{i + 1}'
            if self.label_path:
                transformed = transform(image=image, bboxes=bboxes, class_labels=_label_classes)
                self.__save_image(i_new_name, transformed)
                self.__save_label(i_new_name, transformed)
            else:
                transformed = transform(image=image)
                self.__save_image(i_new_name, transformed)


transform = A.Compose([
    A.Blur(blur_limit=4, p=1),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
    A.Rotate(30)], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# transform = A.Compose([
#     A.Blur(blur_limit=4, p=1),
#     A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
#     A.Rotate(30)])
config = {
    "transform": transform,
    "source_path": "D:\\git\\mlutils\\tests\\data\\augmentation\\dataset\\",
    "label_path": "D:\\git\\mlutils\\tests\\data\\augmentation\\labels\\",
    "target_path": "D:\\git\\mlutils\\tests\\data\\augmentation\\tests\\",
    "multiplier": 3
}
obj = Augmentation(config)
obj.augment_images()
