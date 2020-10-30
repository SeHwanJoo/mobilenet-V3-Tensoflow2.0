# cifar10-ResNet

## Training
Trained using two approaches for 300 epochs with cifar-100


## Files
Source Files:
- dataset_util.py : normalization, load_images, build_optimizer    
- train_cifar10.py : train model with cifar-100
- model : mobilent_v3 base layer and models

## Accuracy

### official model
|Model|Validation Accuracy|dataset|
|:------:|:---:|:---:|
|[official-small](https://arxiv.org/abs/1905.02244)|73.69%|cifar100|
|[Mobilenet-V3-small](https://github.com/SeHwanJoo/mobilenet-V3-Tensoflow2.0)|70.91%|cifar100|
|[official-large](https://arxiv.org/abs/1905.02244)|-|cifar100|
|[Mobilenet-V3-large](https://github.com/SeHwanJoo/mobilenet-V3-Tensoflow2.0)|-|cifar100|

### other models
|Model|Validation Accuracy|dataset|
|:------:|:---:|:---:|
|[VGG-16](https://github.com/SeHwanJoo/cifar10-vgg16)|93.15%|cifar10|
|[ResNet-20](https://github.com/SeHwanJoo/cifar10-ResNet-tensorflow)|91.52%|cifar10|
|[ResNet-32](https://github.com/SeHwanJoo/cifar10-ResNet-tensorflow)|92.53%|cifar10|
|[ResNet-44](https://github.com/SeHwanJoo/cifar10-ResNet-tensorflow)|93.16%|cifar10|
|[ResNet-56](https://github.com/SeHwanJoo/cifar10-ResNet-tensorflow)|93.21%|cifar10|
|[ResNet-110](https://github.com/SeHwanJoo/cifar10-ResNet-tensorflow)|93.90%|cifar10|
|[Mobilenet-V3-small](https://github.com/SeHwanJoo/mobilenet-V3-Tensoflow2.0)|70.91%|cifar100|
|[Mobilenet-V3-small](https://github.com/SeHwanJoo/mobilenet-V3-Tensoflow2.0)|-|cifar100|
