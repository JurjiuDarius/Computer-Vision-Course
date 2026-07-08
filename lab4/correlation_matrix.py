import cv2
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, "images")

NUM_BINS = 64 

image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
loaded_images = {f: cv2.imread(os.path.join(images_dir, f)) for f in image_files}

comparison_methods = [
    ("CORRELATION", cv2.HISTCMP_CORREL),
    ("CHI-SQUARE", cv2.HISTCMP_CHISQR),
    ("INTERSECTION", cv2.HISTCMP_INTERSECT),
    ("BHATTACHARYYA", cv2.HISTCMP_BHATTACHARYYA),
]

def compute_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [NUM_BINS, NUM_BINS], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

histograms = {f: compute_histogram(img) for f, img in loaded_images.items()}
num_images = len(image_files)
thumb_size, cell_size = 60, 70
grid_size = num_images * cell_size + 80

for method_name, method_id in comparison_methods:
    scores = np.array([[cv2.compareHist(histograms[f1], histograms[f2], method_id)
                        for f2 in image_files] for f1 in image_files])
    normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

    result = np.ones((grid_size, grid_size, 3), dtype=np.uint8) * 255

    for i, f in enumerate(image_files):
        thumb = cv2.resize(loaded_images[f], (thumb_size, thumb_size))
        result[70 + i*cell_size : 70 + i*cell_size + thumb_size, 5:5+thumb_size] = thumb
        result[5:5+thumb_size, 70 + i*cell_size : 70 + i*cell_size + thumb_size] = thumb

    for i in range(num_images):
        for j in range(num_images):
            x, y = 70 + j*cell_size, 70 + i*cell_size
            cv2.rectangle(result, (x, y), (x+thumb_size, y+thumb_size), (0, int(normalized[i,j]*255), 0), -1)
            text = f"{scores[i,j]:.2f}" if abs(scores[i,j]) < 100 else f"{scores[i,j]:.0f}"
            ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
            cv2.putText(result, text, (x + (thumb_size-ts[0])//2, y + (thumb_size+ts[1])//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255) if normalized[i,j] > 0.5 else (0,0,0), 1)

    cv2.imshow(method_name, result)

cv2.waitKey(0)
cv2.destroyAllWindows()
