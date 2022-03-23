import os

import cv2

from mlutils.business_rule_exceptions import DirectoryNotFound
from mlutils.file_ops.futils import make_dir, copy_file
from mlutils.image_ops.iutils import check_minimum_resolution


def filter_dataset_by_dimension(source_path, target_path, min_width=320, min_height=320):
    """
    This is which filter image dataset by its dimension and remove images which are smaller than the minimum width and height.
    If no minimum width or height is provided, the resolution of 320x320 will be considered as minimum resolution.
    Args:
        source_path: path for source directory.
        target_path: path of target directory
        min_width: The minimum width for valid image.
        min_height: the minimum height for valid image.
    """
    make_dir(target_path)
    if os.path.exists(source_path):
        valid_images = []
        for dir_path, _, filenames in os.walk(source_path):
            for file in filenames:
                image_path = os.path.abspath(os.path.join(dir_path, file))
                image = cv2.imread(image_path, 0)
                if check_minimum_resolution(image, min_width, min_height):
                    valid_images.append(file)
        copy_file(source_path, target_path, valid_images)
    else:
        raise DirectoryNotFound(f'Unable to find source directory')
