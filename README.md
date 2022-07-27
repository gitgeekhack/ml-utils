# machine-learning-utils

----
While working on Machine Learning Projects, the developer used many of the same utilities on various projects throughout
time, such as Data Operation, Image Operation, and File Operation. MLUtils library offers such a utility to developers
for effective machine learning project development.

## Installation Instructions

```
pip install git+https://github.com/gitgeekhack/ml-utils.git
```

## Getting Started

---

### Data Operation

Data Operation utility provides functionality like Image Augmentation, Filter the image based on dimension, and Split
the data for Model Training.

#### Example

- #### Filter Images based on Dimension
  This is the method for filtering image datasets based on their dimensions and discarding images that are less than the
  minimum width and height. If no minimum width or height is specified, 320x320 is assumed to be the minimum resolution.

```python
from mlutils.data.filter import filter_images_by_dimension

filter_images_by_dimension('./source_path', './target_path', min_width=640, min_height=640)
```

- #### Split the Image Dataset
  This method takes the source directory path and divides the given data into training, validation, and unseen testing
  datasets according to the splitting ratio specified as an input for each dataset, and stores the results in the
  destination directory. If no splitting ratio passed, 0.7 (70%) Training & 0.3 (30%) Unseen Test is take into account.

Note: **train**, **valid**, **unseen_test** parameter values must be greater than 0 and less than 1. Sum of the train,
valid, unseen_test values must be 1.

```python
from mlutils.data.splitting import split_dataset_from_dir

split_dataset_from_dir('./source_path', './target_path', train=0.7, unseen_test=0.2, valid=0.1)
```

- #### Augmentation

  This utility is designed to augment dataset of Object Detection (For now only Yolo annotation format supported). To
  augment the dataset using this utility following parameters are required:

> 1. Source Path, where your Dataset is located
> 2. Target Path, where Augmented Dataset you would like to save
> 3. Multiplier, How many times an image should be augmented
> 4. Transform, It is a class object of albumentations library which contains augmentation techniques and their
     configuration


> **Note:** Example needs to be added after refactor

### Image Operation

Image Operation module provide functionality specific to image related operation like Fix skew angle, Calculate IOU,
Match Template, Generate Image Dataset Summary, etc.

#### Example

- #### Get Object

  With this method, the detected object is cropped from the original image. It needs three arguments: an image,
  coordinates, and a label.

```python
import cv2
from mlutils.image.cv_helper import get_object

image = cv2.imread('image.png')
coord = [25, 50, 75, 100]

cropped_image = get_object(target_img=image, coordinates=coord, label='Test')
```

- #### Calculate IOU

  Intersection over Union (IoU) is a metric that allows us to evaluate how much image2 is overlapped to image1. Please
  refer below image to understand metrics result.

| <img src="docs/images/iou.png" title="iou-score"/> | If the result is closer to 1, it implies that the two images are more overlapped; if it is closer to 0, it means that there is less overlap. |
|----------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|

- #### Fix Skew Angle

  This method takes a numpy array of image and angle and return corrected skew angle numpy array of image.

```python
import cv2
from mlutils.image.cv_helper import fix_skew

image = cv2.imread('image.png')
angle = 5

fixed_image = fix_skew(image=image, angle=angle)
```

- #### Template Match

  Template Matching is a method for searching and finding a template image in a source image; if the template image is
  found in the source image, it returns true; otherwise, it returns false.

```python
import cv2
from mlutils.image.cv_helper import match_template

source_image = cv2.imread('image.png')
template_image = cv2.imread('template.png')

is_matched = match_template(template=template_image, image=source_image)
```

- #### YoloV5 Algorithm Utility

    - #### Get BBOX by Label

      This method transforms results returned by YoloV5 detect method to a more useful and user-friendly format. It
      takes YoloV5 result as input argument and return list which contains Name, Label, Confidence Score & Bounding Box.

    - #### Dataset Summary

      This method generate a comprehensive summary such as the volume of a certain label in the dataset using the
      data.yaml file. To get understanding of data.yaml file format please refer to
      this [link](https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml). By default, the save parameter
      is set to _False_; to save the summary as a CSV file, set it to _True_.

```python
from mlutils.image.yolov5 import dataset_summary

dataset_summary(data_file='./data.yaml', save=True)
```

#### Sample Output

<img src="docs/images/table.png" title="Dataset Summary Output"/>

### File Operation

Copy the contents of the directory from the source path to the destination path.

#### Example

```python
from mlutils.file.utils import copy_file

copy_file('./source_path', './target_path')
```

### Improvements

- [ ] Recursive Folder should also copy in File Operation
- [ ] Refactor Augmentation Code
- [ ] Add additional parameter **format** to support other format of Object Detection Algorithm
- [ ] Image Operation module needs to be reformatted
