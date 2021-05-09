CONTAINER="as-15"

sudo docker run $CONTAINER
printf "CONTAINER IS RUNNING"
python save_model.py --model yolov4