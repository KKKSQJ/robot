# singshotpose

## 项目结构
```
ObjectDatasetTools: 制作自定义数据集
6DObjectPosePrediction: 算法
```

## 介绍
该项目只要用于实时的目标 6D姿态估计。

在yolov2的基础上提出一种新的CNN架构。仅通过 RGB 图来预测目标的 6D 姿态而不需要多个阶段或者假设。

网络输入： 
- RGB图
- 对应的 3D坐标 在 2D平面的投影标签（1（类别）+ 1 * 2（质心）+ 8 * 2（8个点）+ 2 （x,y范围））