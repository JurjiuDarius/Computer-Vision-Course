import cv2
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, "images")

cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
diamond_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
x_shape_kernel = np.array([[1,0,0,0,1], [0,1,0,1,0], [0,0,1,0,0], [0,1,0,1,0], [1,0,0,0,1]], dtype=np.uint8)

for image_name in ["rectangle.png", "building.png"]:
    original_image = cv2.imread(os.path.join(images_dir, image_name))
    grayscale = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY_INV)

    corners_smoothed = cv2.erode(cv2.dilate(binary_image, cross_kernel), diamond_kernel)
    corners_preserved = cv2.erode(cv2.dilate(binary_image, x_shape_kernel), square_kernel)
    corner_pixels = cv2.absdiff(corners_preserved, corners_smoothed)

    corner_pixels_dilated = cv2.dilate(corner_pixels, None)
    result_with_corners = original_image.copy()
    result_with_corners[corner_pixels_dilated > 0] = [0, 0, 255]

    cv2.imshow(f"Original - {image_name}", original_image)
    cv2.imshow(f"Corners - {image_name}", result_with_corners)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
