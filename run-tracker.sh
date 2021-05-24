CONTAINER="as-15"

sudo docker run $CONTAINER
printf "CONTAINER IS RUNNING"
python3 object_tracker.py --pictures_path ./data/pictures/ --output ./outputs/tracker.avi --model yolov4 --dont_show --info