# yolov4-deepsort
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zmeSTP3J5zu2d5fHgsQC06DyYEYJFXq1?usp=sharing)

Object tracking implemented with YOLOv4, DeepSort, and TensorFlow. YOLOv4 is a state of the art algorithm that uses deep convolutional neural networks to perform object detections. We can take the output of YOLOv4 feed these object detections into Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric) in order to create a highly accurate object tracker.

## Demo of Object Tracker on Persons
<p align="center"><img src="data/helpers/demo.gif"\></p>

## Demo of Object Tracker on Cars
<p align="center"><img src="data/helpers/cars.gif"\></p>

## Getting Started
Navigate to the Yolov4_Deepsort-Person-Tracking folder. Then execute the following commands:

```
sh build-docker-image.sh

sudo docker rung -it as15_my_img

sh run-tracker.sh
``` 
