#install torchvision

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import os


# Data transformations for CIFAR-10
transform_cifar = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# Load CIFAR-10 dataset
trainset_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
trainloader_cifar = DataLoader(trainset_cifar, batch_size=128, shuffle=True, num_workers=2)

testset_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
testloader_cifar = DataLoader(testset_cifar, batch_size=100, shuffle=False, num_workers=2)

print("✅ CIFAR-10 dataset loaded successfully!")


# Ensure TinyImageNet is downloaded
if not os.path.exists("./data/tiny-imagenet-200"):
    !wget -O tiny-imagenet-200.zip http://cs231n.stanford.edu/tiny-imagenet-200.zip
    !unzip -q tiny-imagenet-200.zip -d ./data
    !rm tiny-imagenet-200.zip
    print("✅ TinyImageNet downloaded and extracted.")
else:
    print("✅ TinyImageNet already exists.")

# ImageNet-style transformation (using TinyImageNet)
transform_imagenet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load TinyImageNet dataset
trainset_imagenet = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/train", transform=transform_imagenet)
trainloader_imagenet = DataLoader(trainset_imagenet, batch_size=128, shuffle=True, num_workers=2)

testset_imagenet = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/val", transform=transform_imagenet)
testloader_imagenet = DataLoader(testset_imagenet, batch_size=100, shuffle=False, num_workers=2)

print("✅ TinyImageNet dataset loaded successfully!")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ImageNet models (pretrained)
resnet18 = models.resnet18(pretrained=True).to(device)
resnet34 = models.resnet34(pretrained=True).to(device)
resnet50 = models.resnet50(pretrained=True).to(device)
alexnet = models.alexnet(pretrained=True).to(device)

# Ensure model directory exists
os.makedirs("pretrained_models", exist_ok=True)

# Save pretrained models
torch.save(resnet18.state_dict(), "pretrained_models/resnet18.pth")
torch.save(resnet34.state_dict(), "pretrained_models/resnet34.pth")
torch.save(resnet50.state_dict(), "pretrained_models/resnet50.pth")
torch.save(alexnet.state_dict(), "pretrained_models/alexnet.pth")

print("✅ ImageNet models saved.")


from torchvision.models.resnet import BasicBlock  # ADD THIS LINE

# Define a simple ResNet model for CIFAR-10 (fixing previous errors)
class ResNet_CIFAR(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(ResNet_CIFAR, self).__init__()
        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlock, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            downsample = None
            if stride != 1 or self.in_planes != planes:  # When size or channels change
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes)
                )
            
            layers.append(block(self.in_planes, planes, stride, downsample))
            self.in_planes = planes  # Update channels
            
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# Initialize models
resnet20 = ResNet_CIFAR([3, 3, 3]).to(device)
resnet32 = ResNet_CIFAR([5, 5, 5]).to(device)
resnet56 = ResNet_CIFAR([9, 9, 9]).to(device)

# Initialize VGG model
vgg16 = models.vgg16()
vgg16.classifier[6] = nn.Linear(4096, 10)  # Modify for CIFAR-10
vgg16 = vgg16.to(device)

# Wide ResNet (WRN)
from torchvision.models import wide_resnet50_2
wrn = wide_resnet50_2()
wrn.fc = nn.Linear(2048, 10)  # Adjust for CIFAR-10
wrn = wrn.to(device)


# Training function
def train_model(model, trainloader, testloader, epochs=10, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}, Acc: {acc:.2f}%")

# Train CIFAR-10 models
for model, name in zip([resnet20, resnet32, resnet56, vgg16, wrn], 
                        ["resnet20", "resnet32", "resnet56", "vgg16", "wrn"]):
    print(f"\nTraining {name} on CIFAR-10...")
    train_model(model, trainloader_cifar, testloader_cifar)

    # Save the trained model
    torch.save(model.state_dict(), f"pretrained_models/{name}.pth")
    print(f"{name} saved.")

print("\n✅ All CIFAR-10 models saved.")


