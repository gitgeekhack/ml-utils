def check_minimum_dimension(image, min_width=320, min_height=320):
    """
    This is which check if the image is bigger than the minimum dimension.
    If no minimum width or height is provided, the resolution of 320x320 will be considered as minimum resolution.
    Args:
        image: <numpy.ndarray> the input image to be checked.
        min_width: <int> the minimum width for valid image.
        min_height: <int> the minimum height for valid image.
    Returns:
        <Boolean> or None: True If image is bigger than the minimum dimensions else False and None if image is None
    """
    if image is not None:
        if image.shape[0] >= min_height and image.shape[1] >= min_width:
            return True
        return False
    return None
