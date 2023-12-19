from torchsummary import summary
import torchvision.models as models
import torch.nn as nn
import torch

# Build ResNet50 model
resnet50 = models.resnet50(weights=None)
num_ftrs = resnet50.fc.in_features
print(num_ftrs)
# Replace the output layer with a FC layer of 1 node and Sigmoid activation
resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50 = resnet50.to(device)

# Display the model structure using torchsummary
summary(resnet50, (3, 224, 224))  # Input image dimensions: (3, 224, 224)
