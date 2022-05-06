# A ConvNet for the 2020s

## 目录

* [1. 简介](#1-简介)
* [2. 数据集和复现精度](#2-数据集和复现精度)
* [3. 准备数据与环境](#3-准备数据与环境)
   * [3.1 准备环境](#31-准备环境)
   * [3.2 准备数据](#32-准备数据)
* [4. 开始使用](#4-开始使用)
   * [4.1 模型训练](#41-模型训练)
   * [4.2 模型评估](#42-模型评估)
   * [4.3 模型预测](#43-模型预测)
   * [4.4 模型导出](#44-模型导出)
* [5. 代码结构](#5-代码结构)
* [6. 自动化测试脚本](#6-自动化测试脚本)
* [7. License](#7-license)
* [8. 参考链接与文献](#8-参考链接与文献)

## 1. 简介

这是一个 PaddlePaddle 实现的 ConvNeXt。

<p align="center">
<img src="https://user-images.githubusercontent.com/8370623/148624004-e9581042-ea4d-4e10-b3bd-42c92b02053b.png" width=100% height=100% 
class="center">
</p>

**论文:** [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

**参考repo:** [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)

在此非常感谢`HannaMao`贡献的[ConvNeXt](https://github.com/facebookresearch/ConvNeXt)，提高了本repo复现论文的效率。


## 2. 数据集和复现精度

数据集为ImageNet，训练集包含1281167张图像，验证集包含50000张图像。

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

您可以从[ImageNet 官网](https://image-net.org/)申请下载数据。

| 模型      | top1 acc (参考精度) | top1 acc (复现精度) | 权重 \| 训练日志 |
|:---------:|:------:|:----------:|:----------:|
| convnext_tiny | 0.821   | 0.821   | checkpoint-best.pd \| log.txt |

权重及训练日志下载地址：[百度网盘](https://pan.baidu.com/s/1zJENrWFRXSFwIPdxt-QMIA?pwd=jsqp)


## 3. 准备数据与环境


### 3.1 准备环境

硬件和框架版本等环境的要求如下：

- 硬件：4 * RTX3090
- 框架：
  - PaddlePaddle >= 2.2.0

* 安装paddlepaddle

```bash
# 需要安装2.2及以上版本的Paddle，如果
# 安装GPU版本的Paddle
pip install paddlepaddle-gpu==2.2.0
# 安装CPU版本的Paddle
pip install paddlepaddle==2.2.0
```

更多安装方法可以参考：[Paddle安装指南](https://www.paddlepaddle.org.cn/)。

* 下载代码

```bash
git clone https://github.com/flytocc/ConvNeXt-paddle.git
cd ConvNeXt-paddle
```

* 安装requirements

```bash
pip install -r requirements.txt
```

### 3.2 准备数据

如果您已经ImageNet1k数据集，那么该步骤可以跳过，如果您没有，则可以从[ImageNet官网](https://image-net.org/download.php)申请下载。


## 4. 开始使用


### 4.1 模型训练

* 单机多卡训练

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus="0,1,2,3" \
    main.py \
    --model convnext_tiny --drop_path 0.1 \
    --batch_size 128 --lr 4e-3 --accum_iter 8 \
    --warmup_epochs 20 \
    --model_ema --model_ema_eval --dist_eval \
    --data_path /path/to/imagenet/ \
    --cls_label_path_train /path/to/train_list.txt \
    --cls_label_path_val /path/to/val_list.txt \
    --output_dir output/convnext_tiny
```

ps: 如果未指定`cls_label_path_train`/`cls_label_path_val`，会读取`data_path`下train/val里的图片作为train-set/val-set。


部分训练日志如下所示。

```
[11:46:22.948892] Epoch: [96]  [ 840/2502]  eta: 0:15:25  lr: 0.003310  loss: 3.6854 (3.5704)  time: 0.5759  data: 0.0005
[11:46:33.860486] Epoch: [96]  [ 860/2502]  eta: 0:15:14  lr: 0.003310  loss: 3.6475 (3.5700)  time: 0.5454  data: 0.0005
```

### 4.2 模型评估

``` shell
python eval.py \
    --model convnext_tiny \
    --batch_size 128 \
    --data_path /path/to/imagenet/ \
    --cls_label_path_val /path/to/val_list.txt \
    --resume $TRAINED_MODEL
```

ps: 如果未指定`cls_label_path_val`，会读取`data_path`/val里的图片作为val-set。

### 4.3 模型预测

```shell
python infer.py \
    --model convnext_tiny \
    --infer_imgs ./demo/ILSVRC2012_val_00020010.JPEG \
    --resume $TRAINED_MODEL
```

<div align="center">
    <img src="./demo/ILSVRC2012_val_00020010.JPEG" width=300">
</div>

最终输出结果为
```
[{'class_ids': [178, 211, 85, 236, 246], 'scores': [0.8764122724533081, 0.0005400953232310712, 0.00053271499928087, 0.00046646789996884763, 0.0004493726301006973], 'file_name': './demo/ILSVRC2012_val_00020010.JPEG', 'label_names': ['Weimaraner', 'vizsla, Hungarian pointer', 'quail', 'Doberman, Doberman pinscher', 'Great Dane']}]
```
表示预测的类别为`Weimaraner（魏玛猎狗）`，ID是`178`，置信度为`0.8764122724533081`。

### 4.4 模型导出

```shell
python export_model.py \
    --model convnext_tiny \
    --output_dir /path/to/save/export_model/ \
    --resume $TRAINED_MODEL

python inference.py \
    --model_file /path/to/save/export_model/output/model.pdmodel \
    --params_file /path/to/save/export_model/output/model.pdiparams \
    --input_file ./demo/ILSVRC2012_val_00020010.JPEG
```

输出结果为
```
[{'class_ids': [178, 211, 85, 236, 246], 'scores': [0.876124918460846, 0.0005408982397057116, 0.0005338677437976003, 0.0004670215421356261, 0.0004502409719862044], 'file_name': './demo/ILSVRC2012_val_00020010.JPEG', 'label_names': ['Weimaraner', 'vizsla, Hungarian pointer', 'quail', 'Doberman, Doberman pinscher', 'Great Dane']}]
```
表示预测的类别为`Weimaraner（魏玛猎狗）`，ID是`178`，置信度为`0.876124918460846`。与infer.py结果的误差在正常范围内。


## 5. 代码结构

```
├── models.py
├── demo
├── engine.py
├── eval.py
├── export_model.py
├── inference.py
├── infer.py
├── main.py
├── README.md
├── requirements.txt
├── test_tipc
└── util
```


## 6. 自动化测试脚本

**详细日志在test_tipc/output**

TIPC: [TIPC: test_tipc/README.md](./test_tipc/README.md)

首先安装auto_log，需要进行安装，安装方式如下：
auto_log的详细介绍参考https://github.com/LDOUBLEV/AutoLog。
```shell
git clone https://github.com/LDOUBLEV/AutoLog
cd AutoLog/
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-1.2.0-py3-none-any.whl
```
进行TIPC：
```bash
bash test_tipc/prepare.sh test_tipc/config/ConvNeXt/convnext_tiny.txt 'lite_train_lite_infer'

bash test_tipc/test_train_inference_python.sh test_tipc/config/ConvNeXt/convnext_tiny.txt 'lite_train_lite_infer'
```
TIPC结果：

如果运行成功，在终端中会显示下面的内容，具体的日志也会输出到`test_tipc/output/`文件夹中的文件中。

```
Run successfully with command - python3.7 main.py --model=convnext_tiny --data_path=./dataset/ILSVRC2012/ --cls_label_path_train=./dataset/ILSVRC2012/train_list.txt --cls_label_path_val=./dataset/ILSVRC2012/val_list.txt --dist_eval    --output_dir=./test_tipc/output/norm_train_gpus_0_autocast_null/convnext_tiny --epochs=2     --batch_size=8 !
Run successfully with command - python3.7 eval.py --model=convnext_tiny --data_path=./dataset/ILSVRC2012/ --cls_label_path_val=./dataset/ILSVRC2012/val_list.txt --resume=./test_tipc/output/norm_train_gpus_0_autocast_null/convnext_tiny/checkpoint-latest.pd !
Run successfully with command - python3.7 export_model.py --model=convnext_tiny --resume=./test_tipc/output/norm_train_gpus_0_autocast_null/convnext_tiny/checkpoint-latest.pd --output=./test_tipc/output/norm_train_gpus_0_autocast_null !
Run successfully with command - python3.7 inference.py --use_gpu=True --use_tensorrt=False --precision=fp32 --model_file=./test_tipc/output/norm_train_gpus_0_autocast_null/model.pdmodel --batch_size=1 --input_file=./dataset/ILSVRC2012/val  --params_file=./test_tipc/output/norm_train_gpus_0_autocast_null/model.pdiparams > ./test_tipc/output/python_infer_gpu_usetrt_False_precision_fp32_batchsize_1.log 2>&1 !
......
```

* 更多详细内容，请参考：[TIPC测试文档](./test_tipc/README.md)。


## 7. License

This project is released under the MIT license.

## 8. 参考链接与文献
1. A ConvNet for the 2020s: https://arxiv.org/abs/2201.03545
2. ConvNeXt: https://github.com/facebookresearch/ConvNeXt
