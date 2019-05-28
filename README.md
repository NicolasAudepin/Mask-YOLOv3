# Mask-YOLOv3

Mask-yolov3 inspired by Mask-RCNN.

The network can be divided into two parts, one is YOLOv3 and the other is Mask branch.

Training: first train the YOLOv3 network, fine-tune tiny YOLOv3. Once the training is complete, train the entire network together.

Mask-YOLOv3 can be easily switching backbone to satisfy the speed and precision trade-offs.

It is more accurate to do instance segmentation in the boundary box obtained by yolov3 than to directly do instance segmentation on the whole image.

✅ data set (object detection and segmentation mark)

### Implement YOLOv3 by Keras.

Refactoring code for extensibility。

✅ read data set

✅ yolo training part of branches

✅ yolo testing part

### YOLOv3 + FCN branch

[ ] network structure adds FCN branch

[ ] the mask-yolo training part

[ ] mask-yolo detection part



参考Mask-RCNN来实现Mask-YOLOv3。

网络可以分为两个部分，一部分是YOLOv3，另一部分是Mask分支。

训练：首先训练YOLOv3网络，fine-tune tiny yolov3。训练完成以后，一起训练整个网络。

本项目实现的Mask-YOLOv3可以任意切换backbone以满足速度和精度的权衡。

同时在yolov3得到的边界框中做实例分割相比于直接在整个图像上做实例分割会有更高的精度。

✅ 数据集 (目标检测和实例分割标注)

### 使用Keras实现YOLOv3。

✅ 读取数据集部分

✅ yolo分支训练部分

✅ yolo检测部分

### 参考Mask-RCNN修改YOLOv3网络框架。

[ ] 网络结构添加FCN分支

[ ] mask-yolo训练部分

[ ] mask-yolo检测部分



