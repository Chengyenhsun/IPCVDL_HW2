import torch
import torchvision.models as models
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ResNet50-specific mean and std values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Create a ResNet50 model
resnet50 = models.resnet50(pretrained=False, num_classes=2)
# Load pre-trained weights
resnet50.load_state_dict(
    torch.load("resnet50/with_RE/2resnet50_checkpoint_epoch_20.pt", map_location=device)
)  # Make sure the path is correct
resnet50.to(device)
resnet50.eval()

# Load and preprocess the input image
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

image = Image.open("2.jpg")  # Replace with the path to your image
image = (
    transform(image).unsqueeze(0).to(device)
)  # Add a batch dimension and move to GPU (if available)

# Perform inference using the model
with torch.no_grad():
    outputs = resnet50(image)

# Get class probabilities
probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

# Get the predicted class index
predicted_class = torch.argmax(probabilities).item()

# Load class names for ImageNet
# You may need to replace or adapt this depending on the specific classes your model was trained on
class_names = ["cat", "dog"]  # Replace with your actual class names

# Output results
print("Predicted class: {} ({})".format(class_names[predicted_class], predicted_class))
print("Class probabilities:")
for i, prob in enumerate(probabilities):
    print("{}: {:.2f}%".format(class_names[i], prob * 100))

probs = [prob.item() for prob in probabilities]

# Create a bar chart
plt.figure(figsize=(6, 6))
plt.bar(class_names, probs, alpha=0.7)

# Set chart title and axis labels
plt.title("Probability of each class")
plt.xlabel("Class Name")
plt.ylabel("Probability")

# Display probability values on top of the bars
for i, prob in enumerate(probs):
    plt.text(i, prob, f"{prob:.2f}", ha="center", va="bottom")

# Show the bar chart
plt.xticks(rotation=45)  # Make x-axis labels more readable
plt.tight_layout()
plt.show()
