import glob
import os

from tqdm import tqdm

from mlutils.exceptions import UnsupportedObjectType, DirectoryNotFound
from mlutils.file.utils import copy_file, file_exists


def get_bbox_by_label(results):
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


def read_class_labels(label_path):
    with open(label_path) as file:
        class_labels = [line.rstrip() for line in file]
    return class_labels


def split_dataset_by_labels(image_path, annotation_path, class_labels, target_path=None, save=True):
    file_exists(image_path)
    file_exists(annotation_path)
    images_per_label = {k: set() for k in class_labels}
    for filename in tqdm(os.listdir(annotation_path)):
        file = os.path.join(annotation_path, filename)
        if not filename == 'classes.txt':
            with open(os.path.join(annotation_path, file), "r") as f:
                lines = f.readlines()
                for line in lines:
                    label = int(line[0:2])
                    name = filename[:filename.rfind('.')]
                    image_name = glob.glob(f"{image_path}/{name}*")
                    if image_name:
                        image_name = image_name[0][image_name[0].rfind('/') + 1:]
                        images_per_label[class_labels[label]].add(image_name)
    if save:
        for key, value in images_per_label.items():
            copy_file(source_path=image_path, target_path=f'{target_path}/{key}', files=list(value))
    return images_per_label

def summary(data_file):
    with open(data_file, "r") as f:
        lines = f.readlines()

summary(data_file="/home/heli/Desktop/git/maruti-ocr/data/driving_license/new/data.yaml")