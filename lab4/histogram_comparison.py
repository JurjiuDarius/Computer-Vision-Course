import cv2
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, "images")

image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
loaded_images = {f: cv2.imread(os.path.join(images_dir, f)) for f in image_files}
query_image_name = "img02.jpg"

comparison_methods = [
    ("CORRELATION", cv2.HISTCMP_CORREL, True),       # 1 = identical, higher = better
    ("CHI-SQUARE", cv2.HISTCMP_CHISQR, False),       # 0 = identical, lower = better
    ("INTERSECTION", cv2.HISTCMP_INTERSECT, True),   # higher = better
    ("BHATTACHARYYA", cv2.HISTCMP_BHATTACHARYYA, False),  # 0 = identical, lower = better
]

def compute_histogram(image, num_bins):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv_image], [0, 1], None, [num_bins, num_bins], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, 0, 1, cv2.NORM_MINMAX)
    return histogram

def normalize_score(score, reference, higher_is_better):
    if higher_is_better:
        return score / reference if reference != 0 else 0
    else:
        return 1.0 / (1.0 + score)

for num_bins in [256, 64, 32]:
    print(f"\n{'='*60}")
    print(f"COLOR REDUCTION: {num_bins} bins")
    print(f"(each bin covers {256//num_bins} color values)")
    print(f"{'='*60}")

    histograms = {f: compute_histogram(img, num_bins) for f, img in loaded_images.items()}
    query_histogram = histograms[query_image_name]

    for method_name, method_id, higher_is_better in comparison_methods:
        self_comparison = cv2.compareHist(query_histogram, query_histogram, method_id)

        print(f"\n{method_name}")
        print(f"  Reference Q vs Q: {self_comparison:.6f}")
        print(f"  {'Image':<12} {'Raw Score':<14} {'Normalized':<10}")
        print(f"  {'-'*40}")

        scores = []
        for filename, hist in histograms.items():
            raw_score = cv2.compareHist(query_histogram, hist, method_id)
            normalized = normalize_score(raw_score, self_comparison, higher_is_better)
            scores.append((filename, raw_score, normalized))

        scores = [s for s in scores if s[0] != query_image_name]
        scores.sort(key=lambda x: x[2], reverse=True)

        for filename, raw, norm in scores[:5]:
            print(f"  {filename:<12} {raw:<14.6f} {norm:<10.4f}")
