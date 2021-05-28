# yolov4-deepsort

## Getting Started
Open the terminal and navigate to the Yolov4_Deepsort-Person-Tracking folder. Then execute the following commands:

```
sh build-docker-image.sh

sudo docker run -it as15_my_img

sh run-tracker.sh
``` 

## How to export the images from nano to the host

``` 
// get container id
sudo docker ps -a 

// User example - p15
sudo docker cp {CONTAINERID}:/Yolov4_Deepsort-Person-Tracking/outputs /home/{USER}
```