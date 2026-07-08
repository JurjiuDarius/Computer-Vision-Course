import cv2
import numpy as np

hollow_shapes = np.zeros((300, 400), dtype=np.uint8)
cv2.circle(hollow_shapes, (100, 150), 50, 255, 2)
cv2.rectangle(hollow_shapes, (200, 100), (350, 200), 255, 2)

filled_circle = np.zeros_like(hollow_shapes)
filled_rectangle = np.zeros_like(hollow_shapes)
cv2.circle(filled_circle, (100, 150), 50, 255, -1)
cv2.rectangle(filled_rectangle, (200, 100), (350, 200), 255, -1)
inside_mask = filled_circle | filled_rectangle

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
filled_shapes = hollow_shapes.copy()

for _ in range(50):
    dilated = cv2.dilate(filled_shapes, kernel)
    filled_shapes = cv2.bitwise_and(dilated, inside_mask)

cv2.imshow("Original Hollow Shapes", hollow_shapes)
cv2.imshow("Filled Shapes (Morphological)", filled_shapes)
cv2.waitKey(0)
cv2.destroyAllWindows()
