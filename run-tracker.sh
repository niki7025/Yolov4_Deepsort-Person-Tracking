CONTAINER="as-15"

printf "CONTAINER IS RUNNING"

sudo docker exec -it my_container RUN OPENBLAS_CORETYPE=ARMV8 python3 object_tracker.py --framework tflite --pictures_path ./data/pictures/ --output ./outputs/tracker.avi --model yolov4 --dont_show --info --is_output_pictures --weights ./checkpoints/yolov4-416-fp32.tflite --tiny


# Copy the frames from container to the host volume.
# docker cp $CID:/file/path/within/container /host/path/target