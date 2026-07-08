import cv2
import numpy as np

hollow_shapes = np.zeros((300, 400), dtype=np.uint8)
cv2.circle(hollow_shapes, (100, 150), 50, 255, 2)
cv2.rectangle(hollow_shapes, (200, 100), (350, 200), 255, 2)

inverted = cv2.bitwise_not(hollow_shapes)
height, width = inverted.shape

marker = np.zeros((height, width), dtype=np.uint8)
marker[0, :] = inverted[0, :]
marker[height-1, :] = inverted[height-1, :]
marker[:, 0] = inverted[:, 0]
marker[:, width-1] = inverted[:, width-1]

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

while True:
    previous = marker.copy()
    marker = cv2.dilate(marker, kernel)
    marker = cv2.bitwise_and(marker, inverted)
    if np.array_equal(marker, previous):
        break

inside_regions = cv2.bitwise_and(inverted, cv2.bitwise_not(marker))
filled_shapes = cv2.bitwise_or(hollow_shapes, inside_regions)

cv2.imshow("Original Hollow Shapes", hollow_shapes)
cv2.imshow("Final Filled Shapes", filled_shapes)
cv2.waitKey(0)
cv2.destroyAllWindows()
