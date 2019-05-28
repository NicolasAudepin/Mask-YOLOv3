# Mask-YOLOv3

参考Mask-RCNN来实现Mask-YOLOv3。

✅ 使用Pytorch实现YOLOv3检测器部分。

发现Pytorch目前的资料比较少，改用Tensorflow和Keras实现该项目。

网络可以分为两个部分，一部分是YOLOv3，另一部分是Mask分支。

训练：首先训练YOLOv3网络，fine-tune tiny yolov3。训练完成以后，一起训练整个网络。

✅ 数据集 (目标检测和实例分割标注)

### 使用Keras实现YOLOv3。

✅ 读取数据集部分

✅ yolo分支训练部分

✅ yolo检测部分

### 参考Mask-RCNN修改YOLOv3网络框架。

[ ] 网络结构添加FCN分支

[ ] mask-yolo训练部分

[ ] mask-yolo检测部分



