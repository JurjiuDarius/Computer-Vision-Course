# Computer Vision

Lab work and a capstone object-detection demo from my MSc computer vision course — classical image processing built up with OpenCV/NumPy, then modern deep detection with Faster R-CNN.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)

## Labs

| Lab | Topic | Highlights |
|-----|-------|-----------|
| [`lab2/`](lab2/) | Compositing & morphology | Logo overlay with alpha blending; contour filling with morphological cleanup |
| [`lab3/`](lab3/) | Feature detection | Harris corner detection |
| [`lab4/`](lab4/) | Histograms & similarity | Colour-histogram comparison and correlation-matrix image matching |
| [`lab5/`](lab5/) | Shape detection | Hough transform for lines and circles |

## Faster R-CNN demo — [`faster_rcnn/`](faster_rcnn/)

Object detection with a pretrained Faster R-CNN over images and video (`demo.py`), plus `R-CNN_Evolution.pptx` — a walkthrough of the R-CNN → Fast R-CNN → Faster R-CNN lineage. Sample input/output included (`sample.jpg` → `sample_detected.jpg`, `sample_video.mp4`).

## Run

```bash
pip install opencv-python numpy torch torchvision
python lab3/corner_detection.py       # any lab script runs standalone
python faster_rcnn/demo.py
```

## Tech stack

Python · OpenCV · NumPy · PyTorch / torchvision (Faster R-CNN)

— Darius Jurjiu
