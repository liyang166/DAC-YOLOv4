#rm test_dnn_out.avi

./darknet detector demo ./cfg/coco.data ./cfg/yolov4.cfg ./yolov4.weights rtsp://admin:lab346348@115.24.161.81:80 -i 0 -thresh 0.25



