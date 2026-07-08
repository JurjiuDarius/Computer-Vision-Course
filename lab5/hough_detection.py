import cv2
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, "images")

# Line detection
sudoku_image = cv2.imread(os.path.join(images_dir, "sudoku.jpg"))
sudoku_gray = cv2.cvtColor(sudoku_image, cv2.COLOR_BGR2GRAY)
sudoku_edges = cv2.Canny(sudoku_gray, 50, 150)

for threshold in [100, 150, 200]:
    detected_lines = cv2.HoughLines(sudoku_edges, 1, np.pi/180, threshold)
    result_image = sudoku_image.copy()
    if detected_lines is not None:
        for line in detected_lines:
            rho, theta = line[0]
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            x0, y0 = cos_theta * rho, sin_theta * rho
            point1 = (int(x0 + 1000*(-sin_theta)), int(y0 + 1000*cos_theta))
            point2 = (int(x0 - 1000*(-sin_theta)), int(y0 - 1000*cos_theta))
            cv2.line(result_image, point1, point2, (0, 0, 255), 2)
        print(f"threshold={threshold}: {len(detected_lines)} lines")
    cv2.imshow(f"Lines threshold={threshold}", result_image)

# Segment detection
road_image = cv2.imread(os.path.join(images_dir, "road.jpg"))
road_gray = cv2.cvtColor(road_image, cv2.COLOR_BGR2GRAY)
road_edges = cv2.Canny(road_gray, 50, 150)

for min_line_length in [20, 50, 100]:
    detected_segments = cv2.HoughLinesP(road_edges, 1, np.pi/180, 50, minLineLength=min_line_length, maxLineGap=10)
    result_image = road_image.copy()
    if detected_segments is not None:
        for segment in detected_segments:
            x1, y1, x2, y2 = segment[0]
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print(f"minLineLength={min_line_length}: {len(detected_segments)} segments")
    cv2.imshow(f"Segments minLen={min_line_length}", result_image)

# Circle detection
coins_image = cv2.imread(os.path.join(images_dir, "coins.jpg"))
coins_gray = cv2.cvtColor(coins_image, cv2.COLOR_BGR2GRAY)
coins_blurred = cv2.medianBlur(coins_gray, 5)

for accumulator_threshold in [20, 30, 50]:
    detected_circles = cv2.HoughCircles(coins_blurred, cv2.HOUGH_GRADIENT, 1, 30, param1=50, param2=accumulator_threshold, minRadius=10, maxRadius=100)
    result_image = coins_image.copy()
    if detected_circles is not None:
        for circle in np.uint16(np.around(detected_circles[0])):
            center_x, center_y, radius = circle
            cv2.circle(result_image, (center_x, center_y), radius, (0, 255, 0), 2)
        print(f"param2={accumulator_threshold}: {len(detected_circles[0])} circles")
    cv2.imshow(f"Circles param2={accumulator_threshold}", result_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
