# DAC-YOLOv4

## 1.Reference
- **Darknet:** [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
- **TensorRT:** [https://github.com/Tianxiaomo/pytorch-YOLOv4/](https://github.com/Tianxiaomo/pytorch-YOLOv4/)

## 2.Code
### (1). DAC-YOLOv4-master
- dac-yolov4 are .cfg, .names and .weights files 
- You can run it by download the DAC-YOLOv4 base on darknet folder
- CMake & build
```
./strawberry_disease_test.sh
```

### (2).Xavier/Nano
- Environment：
```
Ubantu 18.04
TensorRt 7.0
Python 3.8
```
- inference
```
python trt.py dac-yolov4.trt test_img.jpg 416 416
```

## 3.Dataset
- For more dataset link：https://pan.baidu.com/s/1dWt8bpT5R4AAgfC6cQzMWA?pwd=0520 Extraction code：0520. If there is any quesiton, please contact by email:liyang_taas@126.com, thanks.
