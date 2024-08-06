import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

"""contains models for experimentation."""
# Define a simple neural network
class SimpleNet(nn.Module):
    """A very simple model, worked great on the smaller sized images, lacks capacity for the bigger one."""
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Since the images are grayscale, we have 1 input channel
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # The image size after the pooling layer will be 24x24
        self.fc = nn.Linear(16 * 24 * 24, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(torch.flatten(x, 1))
        x = self.fc(x)
        # x = self.sigmoid(x) # use sigmoid depending on the loss function
        return x
class SimpleFCNet(nn.Module):
    """A very simple model, not using convolutional layers at all, instead, just uses fully  connected layers"""

    def __init__(self):
        super(SimpleFCNet, self).__init__()
        # Since the images are grayscale, we have 1 input channel
        self.fc1 = nn.Linear(48*48, 5)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        x = x.view(-1, 48 * 48)  # Flatten the input and ensure it matches the fc1 input size
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes=1):
        super(ModifiedResNet18, self).__init__()
        # Load a pre-trained ResNet18 model
        self.resnet18 = models.resnet18(pretrained=False)

        # Modify the first convolutional layer to accept 2-channel images
        self.resnet18.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the last fully connected layer to output num_classes classes
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, num_classes)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #return self.softmax(self.resnet18(x))
        return self.resnet18(x)

class IntermidiateNet(nn.Module):
    """An intermidiate sized model"""
    def __init__(self):
        super(IntermidiateNet, self).__init__()
        # Since the images are grayscale, we have 1 input channel
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # The image size after the pooling layer will be 24x24
        self.fc = nn.Linear(32 * 24 * 24, 1)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # x = self.sigmoid(x)
        return x


class EnhancedSimpleNet(nn.Module):
    def __init__(self):
        super(EnhancedSimpleNet, self).__init__()
        # Since the images are grayscale, we have 1 input channel
        self.conv1 = nn.Conv2d(1, 16, 5, padding='same')
        self.conv2 = nn.Conv2d(16, 16, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        # Adding another convolutional layer
        self.conv3 = nn.Conv2d(16, 32, 3, padding='same')
        self.conv4 = nn.Conv2d(32, 32, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        # The image size after the pooling layers will be reduced
        # Assuming the input image size is 48x48, the resulting size will be 12x12
        self.fc1 = nn.Linear(32 * 12 * 12, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.bn2(self.conv4(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

