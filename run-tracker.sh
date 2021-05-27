CONTAINER="as-15"

sudo docker run $CONTAINER
printf "CONTAINER IS RUNNING"
OPENBLAS_CORETYPE=ARMV8 python3 object_tracker.py --framework tflite --pictures_path ./data/pictures/ --output ./outputs/tracker.avi --model yolov4 --dont_show --info

python3 object_tracker.py --pictures_path ./data/pictures/ --output ./outputs/tracker.avi --model yolov4 --dont_show --info