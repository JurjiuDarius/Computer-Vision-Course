import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import cv2
import os
import argparse

pretrained_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
detection_model = fasterrcnn_resnet50_fpn(weights=pretrained_weights)
detection_model.eval()
class_names = pretrained_weights.meta["categories"]

script_dir = os.path.dirname(os.path.abspath(__file__))

def detect(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0

    with torch.no_grad():
        predictions = detection_model([input_tensor])[0]

    for bounding_box, class_label, confidence_score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if confidence_score > 0.5:
            x1, y1, x2, y2 = bounding_box.int().tolist()
            detected_class = class_names[class_label]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{detected_class}: {confidence_score:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def run_image(image_path):
    frame = cv2.imread(image_path)
    result = detect(frame)
    cv2.imshow("Faster R-CNN Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = detect(frame)
        cv2.imshow("Faster R-CNN Video Detection", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faster R-CNN Object Detection")
    parser.add_argument("mode", nargs="?", choices=["image", "video"], default="video", help="Run on image or video")
    args = parser.parse_args()

    if args.mode == "image":
        run_image(os.path.join(script_dir, "sample.jpg"))
    else:
        run_video(os.path.join(script_dir, "sample_video.mp4"))
