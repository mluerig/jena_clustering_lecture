import cv2
import numpy as np
from typing import Tuple
from scipy import stats

def resize_with_pad(image: np.array, 
                    new_shape: Tuple[int, int], 
                    padding_color: Tuple[int] = (255, 255, 255)) -> np.array:
    """
    Maintains aspect ratio and resizes with padding.
    Parameters:
    - image: Image to be resized.
    - new_shape: Expected (width, height) of new image.
    - padding_color: Tuple in BGR of padding color
    
    Returns:
    - image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image


def extract_features(contour, gray, mask):
    """
    Extracts geometric and texture features from a given contour and associated ROI in an image.
    
    Parameters:
    - contour: The contour from which geometric features are extracted.
    - gray: Grayscale image from which the ROI is derived.
    - mask: Binary mask for the ROI, matching dimensions of `gray`.
    
    Returns:
    - A list of extracted features: area, diameter, circularity, mean, variance, skewness, kurtosis.
    """
    # Geometric features
    area = cv2.contourArea(contour)
    diameter = int(cv2.minEnclosingCircle(contour)[1] * 2)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter else 0

    # Extract ROI using bounding rectangle for texture analysis
    x, y, w, h = cv2.boundingRect(contour)
    roi = gray[y:y+h, x:x+w]
    roi_mask = np.logical_not(mask[y:y+h, x:x+w])

    # Create a masked array that ignores the background of the ROI
    masked_array = np.ma.masked_array(roi, mask=roi_mask == 0)

    # Texture features
    mean = np.ma.mean(masked_array)
    variance = np.ma.var(masked_array)
    skewness = stats.skew(masked_array.compressed())
    kurt = stats.kurtosis(masked_array.compressed())

    # Return all features in a list
    return [area, diameter, circularity, mean, variance, skewness, kurt]


def refactor_array(data, threshold=5):
    """
    Refactor an array such that each distinct number, once it has appeared at least 'threshold' times,
    is sequentially mapped to integers starting from 0 based on the order of its first appearance
    after reaching the threshold.

    Parameters:
    data (np.array): The original numpy array with numerical data.
    threshold (int): The minimum number of occurrences before a number is reassigned.

    Returns:
    np.array: A new array where each value meeting the occurrence threshold is replaced by an integer
              starting from 0 based on the order of its first appearance post-threshold.
    """
    # Track first appearances and count occurrences
    occurrence_count = {}
    refactoring_map = {}
    output_array = np.empty_like(data)

    for i, num in enumerate(data):
        # Update occurrence count
        if num in occurrence_count:
            occurrence_count[num] += 1
        else:
            occurrence_count[num] = 1

        # Check if the number meets the threshold for refactoring
        if occurrence_count[num] == threshold:
            if num not in refactoring_map:
                refactoring_map[num] = len(refactoring_map)
        
        # Assign new value if eligible, otherwise assign original value
        if num in refactoring_map:
            output_array[i] = refactoring_map[num]
        else:
            output_array[i] = num  # Assign a placeholder or the original value until refactoring

    return output_array