# Roedeer Recognition

The goal of this implementation is to detect individual roe deer including their gender in camera trap images.

![example image](https://github.com/nhung-huyen-vu/roe-deer-recognition/raw/main/example.jpg)

For object detection the YOLOv7 detector is used. To improve gender classification, a second stage classifier is used to 
predict whether gender is identifiable. The second stage is using a resnet architecture, the implementation is found in the resnet\_classifier subdirectory.

test\_two\_stage.py hooks up the resnet classifier with the yolov7 detector for inference on the test set and for
evaluation using the yolo visualizations.

detect\_two\_stage.py hooks up the resnet classifier with the yolov7 detector for inference on unlabelled data.
