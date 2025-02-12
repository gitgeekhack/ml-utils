import os
import glob
import shutil

import yaml
import termtables
import pandas as pd
from tqdm import tqdm

from mlutils.exceptions import UnsupportedObjectType

__all__ = ['get_bbox_by_label', 'read_label_classes', 'split_dataset_by_labels', 'dataset_summary']


def __copy_file(source_path, target_path, files=None):
    os.makedirs(target_path, exist_ok=True)
    if files is None:
        files = os.listdir(source_path)
    for file in tqdm(files, desc=f"Copying files to {target_path.split('/')[-1]}"):
        shutil.copy(os.path.join(source_path, file), target_path)


def get_bbox_by_label(results):
    """
    This method transforms results returned by yolov5 detect method to a more useful and user-friendly format.
    Args:
        results: <class object> result object returned by yolov5 detect method
    Returns:
        transformed_results: <list> or None transformed results returned by yolov5 (name,label,bbox,conf)

    """
    if results:
        if results.__class__.__name__ == 'Detections':
            transformed_results = []
            for result in results.pandas().xyxy:
                transformed_result = []
                for x in result.values:
                    bbox = tuple(x[:4])
                    conf = x[-3]
                    label = int(x[-2])
                    name = x[-1]
                    transformed_result.append((name, label, bbox, conf))
                transformed_results.append(transformed_result)
            return transformed_results
        raise UnsupportedObjectType('Input object type is not supported')
    return None


def read_label_classes(label_path):
    """
    This method reads class labels from class.txt file.
    Args:
        label_path: <string> path to class label file.
    Returns:
        class_labels: <list> list of class labels
    """
    with open(label_path) as file:
        class_labels = [line.rstrip() for line in file]
    return class_labels


def split_dataset_by_labels(image_path, annotation_path, class_labels, target_path=None, save=False):
    """
    This method splits image dataset by class labels.
    Args:
        image_path: <string> path to image folder.
        annotation_path: <string> path to respective annotation folder.
        class_labels: <list> list of class labels.
        target_path: <string> path to target folder where the sorted images will be saved.
        save: <boolean> Saves images to folder if True.
    Returns:
        images_per_label: <dict> returns set of image name as value and respective labels as key.
    """
    images_per_label = {k: set() for k in class_labels}
    for filename in tqdm(os.listdir(annotation_path), 'splitting dataset'):
        file = os.path.join(annotation_path, filename)
        if not filename == 'classes.txt':
            with open(file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    label = int(line[0:2])
                    name = filename[:filename.rfind('.')]
                    image_name = glob.glob(f"{image_path}/{name}*")
                    if image_name:
                        image_name = image_name[0][image_name[0].rfind('/') + 1:]
                        images_per_label[class_labels[label]].add(image_name)
    if save and target_path:
        for key, value in images_per_label.items():
            __copy_file(source_path=image_path, target_path=f'{target_path}/{key}', files=list(value))
    return images_per_label


def dataset_summary(data_file, save=False):
    """
    This method returns a detailed summary of the data-set with the help of data.yaml file,
    also saves the summary to csv file if save is True
    Args:
        data_file: <string> path to data.yaml file
        save: <boolean> Saves summary to csv file if True.
    """
    with open(data_file, 'r') as stream:
        data = yaml.safe_load(stream)
        class_labels = data['names']
        directory = os.path.dirname(data_file)
        valid_path = directory + data['val'].replace('.', '')
        test_path = directory + data['test'].replace('.', '')
        train_path = directory + data['train'].replace('.', '')
        valid_summary = split_dataset_by_labels(image_path=valid_path,
                                                annotation_path=valid_path.replace('images', 'labels'),
                                                class_labels=class_labels, save=False)
        test_summary = split_dataset_by_labels(image_path=test_path,
                                               annotation_path=test_path.replace('images', 'labels'),
                                               class_labels=class_labels, save=False)
        train_summary = split_dataset_by_labels(image_path=train_path,
                                                annotation_path=train_path.replace('images', 'labels'),
                                                class_labels=class_labels, save=False)
        valid_images = {k: len(v) for k, v in valid_summary.items()}
        test_images = {k: len(v) for k, v in test_summary.items()}
        train_images = {k: len(v) for k, v in train_summary.items()}
        header = ['No', "Label", 'Train', 'Test', 'Valid']
        data = [[i + 1, label, train_images[label], test_images[label], valid_images[label]] for i, label in
                enumerate(class_labels)]
        termtables.print(data, header, style=termtables.styles.rounded_thick)
        if save:
            my_df = pd.DataFrame(data)
            my_df.to_csv(directory + '/summary.csv', header=header, index=False)
