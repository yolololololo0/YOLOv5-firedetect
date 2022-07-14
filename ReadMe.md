# 基于yolov5的火焰识别

## 1、准备工作

### yolov5项目下载

​		下载yolov5项目代码，其链接为：[yolov5项目地址](https://github.com/ultralytics/yolov5)

​		并且在PC机上配置环境，即正常按照requirements安装依赖包，而后根据自身需要下载相应的权重文件(yolov5s、yolov5m、yolov5l、yolov5x)

### 数据集的准备

​		1、根据实际情况可以自身在网上爬取火焰图片

​		2、通过网上的资料下载相关数据集，大部分数据集是无标注的数据集，此处可参考工控-小白的博客[博客地址](https://blog.csdn.net/xiaorongronglove/article/details/116125936)，其中就有部分火焰识别数据集的下载链接

​		如下载的数据集无标注，那么使用lableImg进行标注，且将标注文件的保存格式设置为PascalVOC的类型，即xml格式的label文件，而后通过脚本将标签格式转换为.txt文件，并在文件上添加类别信息和对数据进行归一化。

```python
import os
import xml.etree.ElementTree as ET
from decimal import Decimal
 
dirpath = '/home/jiu/data_change/label_0'  # 原来存放xml文件的目录
newdir = '/home/jiu/data_change/labels'  # 修改label后形成的txt目录
 
if not os.path.exists(newdir):
    os.makedirs(newdir)
 
for fp in os.listdir(dirpath):
 
    root = ET.parse(os.path.join(dirpath, fp)).getroot()
 
    xmin, ymin, xmax, ymax = 0, 0, 0, 0
    sz = root.find('size')
    width = float(sz[0].text)
    height = float(sz[1].text)
    filename = root.find('filename').text
    print(fp)
    with open(os.path.join(newdir, fp.split('.')[0] + '.txt'), 'a+') as f:
        for child in root.findall('object'):  # 找到图片中的所有框
 
            sub = child.find('bndbox')  # 找到框的标注值并进行读取
            sub_label = child.find('name')
            xmin = float(sub[0].text)
            ymin = float(sub[1].text)
            xmax = float(sub[2].text)
            ymax = float(sub[3].text)
            try:  # 转换成yolov的标签格式，需要归一化到（0-1）的范围内
                x_center = Decimal(str(round(float((xmin + xmax) / (2 * width)),6))).quantize(Decimal('0.000000'))
                y_center = Decimal(str(round(float((ymin + ymax) / (2 * height)),6))).quantize(Decimal('0.000000'))
                w = Decimal(str(round(float((xmax - xmin) / width),6))).quantize(Decimal('0.000000'))
                h = Decimal(str(round(float((ymax - ymin) / height),6))).quantize(Decimal('0.000000'))
                print(str(x_center) + ' ' + str(y_center)+ ' '+str(w)+ ' '+str(h))
                #读取需要的标签
                if sub_label.text == 'fire':
                    f.write(' '.join([str(0), str(x_center), str(y_center), str(w), str(h) + '\n']))
            except ZeroDivisionError:
                print(filename, '的 width有问题')

```

​		此处提供本人所使用的火焰数据集，该数据集一共1421张带火焰的图片，并将其分为训练集和测试集，其中训练集1200张，测试集221张；同样的将label也分为训练集和测试集，其图片和其label相对应。

数据集下载地址：[Let's  go](https://download.csdn.net/download/weixin_43482623/21008146)

配置文件修改

1、新建一个.yaml文件，在其中添加：

```python
train: /home/jiu/project/fire_detect/dataset/images/train  # train images 1200 images
val: /home/jiu/project/fire_detect/dataset/images/val  # val images 221 images
test:  # test images (optional)

# Classes
nc: 1  # number of classes
names: [ 'fire' ]  # class names
```

