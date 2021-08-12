import torch
import torch.nn as nn
import torch.nn.functional as F

class AllConvNet(nn.Module):

  """ 
  Following the architecture given in the paper
  Methods
  --------
  forward(x)
    - return prediction tensor
  """
  def __init__(self, image_size, n_classes, **kwargs):

    """
    Parameters
    ----------
    image_size : int
        Number of input dimensions aka side length of image
    n_classes: int
        Number of classes in the dataset
    """
    super(AllConvNet, self).__init__()
    #Constructing the model as per the paper
    self.conv1 = nn.Conv2d(image_size, 96, 3, padding=1)
    self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
    self.conv3 = nn.MaxPool2d(3, stride=2)
    self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
    self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
    self.conv6 = nn.MaxPool2d(3, stride=2)
    self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
    self.conv8 = nn.Conv2d(192, 192, 1)
    self.conv9 = nn.Conv2d(10, 10, 1)

    self.class_conv = nn.Conv2d(192, n_classes, 1)

  #Forward pass of the model
  def forward(self, x):
    """
    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    Returns
    -------
    output : torch.Tensor
        Generated sample
    """
    conv1_out = F.relu(self.conv1(x))
    conv2_out = F.relu(self.conv2(conv1_out))
    conv3_out = F.relu(self.conv3(conv2_out))
    conv4_out = F.relu(self.conv4(conv3_out))
    conv5_out = F.relu(self.conv5(conv4_out))
    conv6_out = F.relu(self.conv6(conv5_out))
    conv7_out = F.relu(self.conv7(conv6_out))
    conv8_out = F.relu(self.conv8(conv7_out))
    class_out = F.relu(self.class_conv(conv8_out))
    pool_out = F.adaptive_avg_pool2d(class_out, 1)
    pool_out.squeeze_(-1)
    pool_out = F.softmax(pool_out.squeeze_(-1))
    return pool_out