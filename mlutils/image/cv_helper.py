import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

from mlutils.exceptions import MissingRequiredParameterException

__all__ = ['get_object', 'calculate_iou', 'get_skew_angel', 'fix_skew', 'match_template']


def get_object(target_img, coordinates, label):
    """
    Parameters:
        target_img <class 'numpy.ndarray'>: The target image which is to be cropped.
        coordinates <class 'list'>: The coordinates of the region to be cropped, a 4-element list containing
                            of x/y (x0, y0, x1, y1) pixel coordinates.
    Returns:
        cropped_img <class 'numpy.ndarray'>: The final image which is cropped with the coordinates provided.
    """
    x0, y0, x1, y1 = [int(x) for x in coordinates]
    w = target_img.shape[0]
    h = target_img.shape[1]
    padd = 0.005
    w_padd = w * padd
    h_padd = h * padd
    x0, y0 = int(x0 - w_padd), int(y0 - w_padd)
    x1, y1 = int(x1 + h_padd), int(y1 + h_padd)
    cropped_img = target_img[y0:y1, x0:x1]
    return {'detected_object': cropped_img, 'label': label}


def calculate_iou(x, y):
    x0 = max(x[0], y[0])
    y0 = max(x[1], y[1])
    x1 = min(x[2], y[2])
    y1 = min(x[3], y[3])
    inter_area = max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)
    box0_area = (x[2] - x[0] + 1) * (x[3] - x[1] + 1)
    box1_area = (y[2] - y[0] + 1) * (y[3] - y[1] + 1)
    iou = inter_area / float(box0_area + box1_area - inter_area)
    return iou


def __find_skew_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    histogram = np.sum(data, axis=1)
    score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
    return histogram, score


def __calculate_skew_angel(image, delta=5, limit=45):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = __find_skew_score(thresh, angle)
        scores.append(score)
    best_angle = angles[scores.index(max(scores))]
    return best_angle


def get_skew_angel(extracted_objects):
    find_skew_angles = [__calculate_skew_angel(extracted_object['detected_object']) for extracted_object in
                        extracted_objects]
    skew_angles = find_skew_angles
    max_skew_angle = np.max(skew_angles)
    min_skew_angle = np.min(skew_angles)
    if min_skew_angle < 0:
        skew_angle = min_skew_angle
    else:
        skew_angle = max_skew_angle

    return skew_angle


def fix_skew(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def match_template(template, image, threshold=0.9):
    """
    Parameters:
        template <class 'numpy.ndarray'> : The source Searched template.
        image <class 'numpy.ndarray'> : The input image where the search is running.
    Returns:
        Boolean: True if the input image matches the template, False otherwise.
    """
    if template is None:
        raise MissingRequiredParameterException('Missing Required input Template Parameter for Template Matching')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    match = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
    if match.all() >= threshold:
        return True
    return False


def apply_bbox_padding(page_dim, input_bbox, x0_pad=0.0, y0_pad=0.0, x1_pad=0.0, y1_pad=0.0):
    """
        applies padding to the input bounding box relative to the page size.
        Parameters:
          page_dim <tuple> : The page dimension of page for relative padding to bounding box.
          bbox <tuple> : The input bounding box to be padded.
          x0_pad <float> : The input percentage relative to page size to be padded to x0
          x1_pad <float> : The input percentage relative to page size to be padded to x1
          y0_pad <float> : The input percentage relative to page size to be padded to y0
          y1_pad <float> : The input percentage relative to page size to be padded to y1
        Returns:
         <tuple> : The output padded bounding box
    """
    x0, y0, x1, y1 = input_bbox
    w = page_dim[2]
    h = page_dim[3]
    x0, y0 = x0 + w * x0_pad, y0 + h * y0_pad
    x1, y1 = x1 + w * x1_pad, y1 + h * y1_pad
    return x0, y0, x1, y1
