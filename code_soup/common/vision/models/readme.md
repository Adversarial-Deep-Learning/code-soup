# Common Models for Vision

List of implemented models
---
1. [AllConvNet](allconvnet.py), [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806)
2. [Network in Network](nin.py), [Network In Network](https://arxiv.org/abs/1312.4400)

# Notes on existing Torchivision models

| Model  | Input sizes |
| ------------- | ------------- |
| AlexNet  |  Pre-trained to work on RGB images of sizes 256 x 256. Input to the first layer is a random crop of size 227 x 227 (not 224 x 224 as mentioned in the paper). The required minimum input size of the model is 227x227.|
| VGG-net  | Pre-trained to work on RGB images of sizes 256 x 256, cropped to 224 x 224. IThe required minimum input size of the model is 224x224. |
| ResNet | Pre-trained to work on RGB images of sizes 256 x 256, cropped to 224 x 244. IThe required minimum input size of the model is 224x224. |
| Inception-v3 | Pre-trained to work on RGB images of sizes 299 x 299. The pre-trained model, with default aux_logits=True, would work for images of size>=299x299 (example: ImageNet), but not for images of size<299x299 (example: CIFAR-10 and MNIST).|

All other pre-trained models require a minimum input size of 224 x 224.
