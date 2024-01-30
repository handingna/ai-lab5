# ai-lab5:多模态情感分析
当代人工智能实验五

## Setup

This implemetation is based on Python3. To run the code, you need the following dependencies:



You can simply run 

```python
pip install -r requirements.txt
```

## Repository structure
We select some important files for detailed description.

```python
|-- data # 提供的数据集解压后
    |-- data/ # 包括所有的训练文本和图片，每个文件按照唯一的guid命名。
    |-- train.txt  # 数据的guid和对应的情感标签
    |-- test_without_label  # 数据的guid和空的情感标签。
|-- model # 保存训练好的模型
    |-- model.pth  # 融合模型
    |-- model_no_text.pth  # 只输入图像
    |-- model_no_image.pth  # 只输入文本
|-- data.py # 数据处理
|-- model.py # 模型结构
|-- main.py # 主函数
|-- requirements.txt
|-- README.md
```

## Run 



## Attribution

Parts of this code are based on the following repositories:

- [ResNet](https://pytorch.org/hub/pytorch_vision_resnet/)

