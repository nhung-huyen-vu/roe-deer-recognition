# Roedeer Recognition

The goal of this implementation is to detect individual roe deer including their gender in camera trap images.

For object detection the YOLOv7 detector is used. To improve gender classification, detections are reclassified
using a resnet architecture, which is found in the resnet\_classifier subdirectory.

test\_two\_stage.py hooks up the resnet classifier with the yolov7 detector for inference on the test set.
