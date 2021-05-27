CONTAINER="as-15"

#sudo docker run $CONTAINER

CID=$(sudo docker run -d $CONTAINER)

printf "CONTAINER IS RUNNING " + $CID

OPENBLAS_CORETYPE=ARMV8 python3 object_tracker.py --framework tflite --pictures_path ./data/pictures/ --output ./outputs/tracker.avi --model yolov4 --dont_show --info

# Copy the frames from container to the host volume.
#docker cp $CID:/file/path/within/container /host/path/target
