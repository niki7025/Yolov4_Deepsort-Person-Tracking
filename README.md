
## Getting Started
To get started, install the proper dependencies either via Anaconda or Pip. I recommend Anaconda route for people using a GPU as it configures CUDA toolkit version for you.

```bash
sh build-docker-image.sh

OPENBLAS_CORETYPE=ARMV8 python3 object_tracker.py --framework tflite --pictures_path ./data/pictures/ --output ./outputs/tracker.avi --model yolov4 --dont_show --info --is_output_pictures --weights ./checkpoints/yolov4-416-fp32.tflite --tiny
```
